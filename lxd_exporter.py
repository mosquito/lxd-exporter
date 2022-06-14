import logging
import os
import re
import ssl
import threading
import time
from abc import ABC
from collections import Counter, defaultdict
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any, Dict, FrozenSet, Iterable, Mapping, NamedTuple, Optional, Union,
)

import aiohttp
import argclass
from aiohttp import web
from aiomisc import entrypoint, threaded_iterable
from aiomisc.service.aiohttp import AIOHTTPService
from aiomisc.service.periodic import PeriodicService
from aiomisc.service.sdwatchdog import SDWatchdogService
from aiomisc_log import LogFormat, basic_config
from yarl import URL


ValueType = Union[int, float]
INF = float("inf")
MINUS_INF = float("-inf")
NaN = float("NaN")


class LxdGroup(argclass.Group):
    url: URL = argclass.Argument(
        default="unix:///var/snap/lxd/common/lxd/unix.socket",
    )
    server_cert: Optional[Path]
    client_cert: Optional[Path]
    client_key: Optional[Path]


class ListenGroup(argclass.Group):
    address: str
    port: int


class LogGroup(argclass.Group):
    level: int = argclass.LogLevel
    format: str = argclass.Argument(
        default=LogFormat.default(),
        choices=LogFormat.choices(),
    )


class CollectorGroup(argclass.Group):
    interval: int
    delay: int = 0
    skip_interface: FrozenSet[str] = argclass.Argument(
        nargs=argclass.Nargs.ONE_OR_MORE, converter=frozenset,
        default="[]",
    )


class Parser(argclass.Parser):
    log = LogGroup(title="Logging options")
    lxd = LxdGroup(title="LXD options")
    http = ListenGroup(
        title="HTTP server options",
        defaults=dict(address="127.0.0.1", port=8080),
    )
    collector = CollectorGroup(
        title="Collector Service options",
        defaults=dict(interval=30),
    )
    pool_size: int = 4


class MetricBase:
    __slots__ = ("name", "label_names", "type", "help")

    def __init__(
        self, *, name: str, labelnames: Iterable[str], help: str = None,
        type: str = "gauge", namespace: str, subsystem: str, unit: str,
    ):
        self.name = "_".join(filter(None, (namespace, subsystem, name, unit)))
        self.help = help
        self.type = type
        self.label_names = frozenset(labelnames)


class Record(NamedTuple):
    metric: MetricBase
    labels: str

    def set(self, value: ValueType) -> None:
        STORAGE.add(self, float(value))


class Metric(MetricBase):
    def labels(self, **kwargs) -> Record:
        labels = []
        for lname in sorted(self.label_names):
            lvalue = kwargs[lname]

            if lvalue is None:
                continue

            lvalue = str(lvalue).replace('"', '\\"')
            labels.append(f"{lname}=\"{lvalue}\"")

        return Record(
            metric=self, labels=",".join(labels),
        )


class Storage:
    metrics: Dict[MetricBase, Dict[Record, ValueType]]
    metrics_ttl: Dict[MetricBase, float]

    def __init__(self, metric_ttl: int = 600):
        self.metrics = defaultdict(dict)
        self.lock = threading.Lock()
        self.metric_ttl = metric_ttl
        self.metrics_ttl = dict()

    def add(self, record: Record, value: ValueType):
        value = float(value)
        metric = record.metric
        self.metrics_ttl[metric] = time.monotonic()

        if metric in self.metrics and record in self.metrics[metric]:
            self.metrics[record.metric][record] = value
            return

        # prevent change when iterating
        with self.lock:
            self.metrics[metric][record] = value

    def __iter__(self):
        with self.lock:
            ts = time.monotonic()
            for metric, records in self.metrics.items():
                ttl = self.metrics_ttl.get(metric)
                if ttl and ttl + self.metric_ttl < ts:
                    continue

                if metric.help:
                    yield f"# HELP {metric.name} {metric.help}\n"

                if metric.type:
                    yield f"# TYPE {metric.name} {metric.type}\n"

                for record, value in records.items():
                    yield "%s{%s} %.6e\n" % (
                        metric.name, record.labels, value,
                    )


STORAGE = Storage()


class MetricsAPI(AIOHTTPService):
    compression: bool = False

    @threaded_iterable
    def provide_metrics(self):
        for line in STORAGE:
            yield line.encode()

    async def metrics(self, request: web.Request):
        response = web.StreamResponse()
        response.content_type = "text/plain; version=0.0.4; charset=utf-8"
        response.enable_chunked_encoding()

        if self.compression:
            response.enable_compression()

        await response.prepare(request)

        async for line in self.provide_metrics():
            await response.write(line)
        await response.write_eof()
        return response

    async def create_application(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/metrics", self.metrics)
        return app


CONTAINER_STATUSES = MappingProxyType({
    "started": 0,
    "stopped": 0,
    "running": 0,
    "cancelling": 0,
    "pending": 0,
    "starting": 0,
    "stopping": 0,
    "aborting": 0,
    "freezing": 0,
    "frozen": 0,
    "thawed": 0,
    "success": 0,
    "failure": 0,
})


class CollectorBase(ABC):
    @staticmethod
    def dehumanize_size(human_value):
        suffixes = ("", "KB", "MB", "GB", "TB", "PB")
        value, suffix = re.match(r"^(\d+)(\wB)?$", human_value).groups()
        return int(value) * (1024 ** suffixes.index(suffix))

    @staticmethod
    def dehumanize_time(human_value):
        suffixes = {
            "d": 86400,
            "h": 3600,
            "m": 60,
            "s": 1.0,
            "ms": 0.001,
            "us": 0.000001,
            "ns": 0.000000001,
        }
        value, suffix = re.match(r"^([\d.]+)(\w+)?$", human_value).groups()
        return float(value) * suffixes[suffix]

    def update(self, *args):
        pass


class ContainerCollector(CollectorBase):
    pass


class ContainerVirtualCollector(CollectorBase):
    pass


class ContainerDeviceCollector(CollectorBase):
    pass


class StorageCollector(CollectorBase):
    pass


class ProfileCollector(CollectorBase):
    pass


def simple_collector(name, unit, help, labels=("container", "location")):
    class SimpleCollector(ContainerCollector):
        METRIC = Metric(
            namespace="lxd",
            subsystem="container",
            name=name,
            unit=unit,
            help=help,
            labelnames=labels,
        )

        def update(self, container: dict, value: str):
            self.METRIC.labels(
                container=container["name"],
                location=container["location"],
            ).set(int(value))

    return SimpleCollector


BootPriorityCollector = simple_collector(
    name="boot",
    unit="autostart_priority",
    help="Container boot autostart priority",
)


class ImageOSCollector(ContainerCollector):
    METRIC = Metric(
        namespace="lxd",
        subsystem="container",
        name="image",
        unit="os",
        help="Container os name image string",
        labelnames=("container", "location", "os"),
    )

    def update(self, container: dict, value: str):
        self.METRIC.labels(
            container=container["name"],
            location=container["location"],
            os=value,
        ).set(1)


class ImageOSReleaseCollector(ContainerCollector):
    METRIC = Metric(
        namespace="lxd",
        subsystem="container",
        name="image",
        unit="os_release",
        help="Container os release image string",
        labelnames=("container", "location", "release"),
    )

    def update(self, container: dict, value: str):
        self.METRIC.labels(
            container=container["name"],
            location=container["location"],
            release=value,
        ).set(1)


class ImageOSVersionCollector(ContainerCollector):
    METRIC = Metric(
        namespace="lxd",
        subsystem="container",
        name="image",
        unit="os_version",
        help="Container os version image string",
        labelnames=("container", "location", "version"),
    )

    def update(self, container: dict, value: str):
        self.METRIC.labels(
            container=container["name"],
            location=container["location"],
            version=value,
        ).set(1)


class ImageOSSerialCollector(ContainerCollector):
    METRIC = Metric(
        namespace="lxd",
        subsystem="container",
        name="image",
        unit="os_serial",
        help="Container os serial image string",
        labelnames=("container", "location", "serial"),
    )

    def update(self, container: dict, value: str):
        self.METRIC.labels(
            container=container["name"],
            location=container["location"],
            serial=value,
        ).set(1)


LimitsCPUCollector = simple_collector(
    name="limits", unit="cpu",
    help="Container cpu limit",
)

LimitsProcessesCollector = simple_collector(
    name="limits", unit="processes",
    help="Container processes limit",
)


class LimitsMemoryCollector(ContainerCollector):
    METRIC = Metric(
        namespace="lxd",
        subsystem="container",
        name="limits",
        unit="memory",
        help="Container limits memory",
        labelnames=("container", "location"),
    )

    def update(self, container: dict, value: str):
        self.METRIC.labels(
            container=container["name"], location=container["location"],
        ).set(self.dehumanize_size(value))


class LimitsCPUEffectiveCollector(ContainerVirtualCollector):
    METRIC = Metric(
        namespace="lxd",
        subsystem="container",
        name="limits",
        unit="cpu_effective",
        help="Container effective cpu limit",
        labelnames=("container", "location"),
    )

    def get_cpu_limit(self, container: dict) -> Union[int, float]:
        cpus = container["expanded_config"].get("limits.cpu")
        allowance = container["expanded_config"].get("limits.cpu.allowance")

        if cpus is None:
            return INF

        value = int(cpus)
        if allowance is not None:
            allowed, period = map(
                self.dehumanize_time, allowance.split("/", 1),
            )
            return allowed / period

        return value

    def update(self, container: dict):
        value = self.get_cpu_limit(container)

        self.METRIC.labels(
            container=container["name"], location=container["location"],
        ).set(value)


class StateCollector(ContainerVirtualCollector):
    SKIP_INTERFACES: FrozenSet[str] = frozenset([])
    METRIC_CPU = Metric(
        namespace="lxd",
        subsystem="container",
        name="state",
        unit="cpu",
        help="Container CPU state",
        labelnames=("container", "location"),
    )

    METRIC_PROCESSES = Metric(
        namespace="lxd",
        subsystem="container",
        name="state",
        unit="processes",
        help="Container running processes",
        labelnames=("container", "location"),
    )

    METRIC_DISK = Metric(
        namespace="lxd",
        subsystem="container",
        name="state",
        unit="disk_usage",
        help="Container disk device statistic",
        labelnames=("container", "location", "device", "pool", "path"),
    )

    METRIC_MEMORY = Metric(
        namespace="lxd",
        subsystem="container",
        name="state",
        unit="memory_usage",
        help="Container memory usage",
        labelnames=("container", "location"),
    )

    METRIC_MEMORY_PEAK = Metric(
        namespace="lxd",
        subsystem="container",
        name="state",
        unit="memory_usage_peak",
        help="Container memory peak usage",
        labelnames=("container", "location"),
    )

    METRIC_SWAP = Metric(
        namespace="lxd",
        subsystem="container",
        name="state",
        unit="swap_usage",
        help="Container swap usage",
        labelnames=("container", "location"),
    )

    METRIC_SWAP_PEAK = Metric(
        namespace="lxd",
        subsystem="container",
        name="state",
        unit="swap_usage_peak",
        help="Container swap peak usage",
        labelnames=("container", "location"),
    )

    METRIC_IF_STATE = Metric(
        namespace="lxd",
        subsystem="container",
        name="state",
        unit="network_interface",
        help="Container interface state",
        labelnames=("container", "location", "device", "state"),
    )

    METRIC_IP = Metric(
        namespace="lxd",
        subsystem="container",
        name="state",
        unit="network_addresses",
        help="Container IP addresses",
        labelnames=("container", "location", "device", "family"),
    )

    METRIC_NETWORK_RX = Metric(
        namespace="lxd",
        subsystem="container",
        name="state",
        unit="network_bytes_rx",
        help="Container network received bytes",
        labelnames=("container", "location", "device"),
    )

    METRIC_NETWORK_TX = Metric(
        namespace="lxd",
        subsystem="container",
        name="state",
        unit="network_bytes_tx",
        help="Container network transmitted bytes",
        labelnames=("container", "location", "device"),
    )

    METRIC_NETWORK_PACKETS_RX = Metric(
        namespace="lxd",
        subsystem="container",
        name="state",
        unit="network_packets_rx",
        help="Container network received packets",
        labelnames=("container", "location", "device"),
    )

    METRIC_NETWORK_PACKETS_TX = Metric(
        namespace="lxd",
        subsystem="container",
        name="state",
        unit="network_packets_tx",
        help="Container network transmitted bytes",
        labelnames=("container", "location", "device"),
    )

    def update(self, container: dict):
        state: dict = container["state"]
        disks = {
            key: value for key, value in container["expanded_devices"].items()
            if value.get("type") == "disk"
        }
        labels = MappingProxyType(
            dict(container=container["name"], location=container["location"]),
        )

        self.METRIC_CPU.labels(**labels).set(state["cpu"]["usage"])
        self.METRIC_PROCESSES.labels(**labels).set(state["processes"])

        for name, usage in state["disk"].items():
            pool = disks.get(name, {}).get("pool")
            path = disks.get(name, {}).get("path")

            self.METRIC_DISK.labels(
                device=name, pool=pool, path=path, **labels
            ).set(usage["usage"])

        self.METRIC_MEMORY.labels(**labels).set(
            state["memory"]["usage"],
        )
        self.METRIC_MEMORY_PEAK.labels(**labels).set(
            state["memory"]["usage_peak"],
        )

        self.METRIC_SWAP.labels(**labels).set(
            state["memory"]["swap_usage"],
        )
        self.METRIC_SWAP_PEAK.labels(**labels).set(
            state["memory"]["swap_usage_peak"],
        )

        if not state["network"]:
            return

        for name, usage in state["network"].items():
            skipping = False
            for skip in self.SKIP_INTERFACES:
                if name.startswith(skip):
                    skipping = True
                    break

            if skipping:
                continue

            self.METRIC_NETWORK_RX.labels(
                device=name, **labels
            ).set(usage["counters"]["bytes_received"])
            self.METRIC_NETWORK_TX.labels(
                device=name, **labels
            ).set(usage["counters"]["bytes_sent"])
            self.METRIC_NETWORK_PACKETS_RX.labels(
                device=name, **labels
            ).set(usage["counters"]["packets_received"])
            self.METRIC_NETWORK_PACKETS_TX.labels(
                device=name, **labels
            ).set(usage["counters"]["packets_sent"])
            self.METRIC_IF_STATE.labels(
                device=name, state=usage["state"], **labels
            ).set(1)

            ip_state = Counter()
            for address in usage["addresses"]:
                ip_state[address["family"]] += 1

            for family, value in ip_state.items():
                self.METRIC_IP.labels(
                    device=name, family=family, **labels
                ).set(value)


class ContainerDiskCollector(ContainerDeviceCollector):
    METRIC = Metric(
        namespace="lxd",
        subsystem="container",
        name="device",
        unit="disk",
        help="Container disk device statistic",
        labelnames=("container", "location", "device", "pool", "path"),
    )

    METRIC_SIZE = Metric(
        namespace="lxd",
        subsystem="container",
        name="device",
        unit="disk_size",
        help="Container disk size device statistic",
        labelnames=("container", "location", "device", "pool", "path"),
    )

    METRIC_LIMITS = Metric(
        namespace="lxd",
        subsystem="container",
        name="device",
        unit="disk_limits",
        help="Container disk size device statistic",
        labelnames=(
            "container", "location", "device", "pool", "path", "read", "write",
        ),
    )

    METRIC_LIMITS_IOPS = Metric(
        namespace="lxd",
        subsystem="container",
        name="device",
        unit="disk_limits_iops",
        help="Container disk size device statistic",
        labelnames=(
            "container", "location", "device", "pool", "path", "read", "write",
        ),
    )

    def update(
        self, container: dict, device_name: str, device: Mapping[str, Any],
    ):
        labels = dict(
            container=container["name"],
            location=container["location"],
            device=device_name,
            pool=device.get("pool"),
            path=device.get("path"),
        )

        self.METRIC.labels(**labels).set(1)

        if "size" in device:
            self.METRIC_SIZE.labels(**labels).set(
                self.dehumanize_size(device["size"]),
            )

        def update_limit(value, **kwargs):
            limits_lables = dict(labels)
            limits_lables.update(kwargs)

            metric = self.METRIC_LIMITS
            if value.endswith("iops"):
                metric = self.METRIC_LIMITS_IOPS
                value = int(value[:-4])
            else:
                value = self.dehumanize_size(value)

            metric.labels(**limits_lables).set(value)

        if device.keys() & {"limits.max", "limits.read", "limits.write"}:
            if "limits.max" in device:
                update_limit(device["limits.max"], read=1, write=1)
            if "limits.read" in device:
                update_limit(device["limits.read"], read=1, write=0)
            if "limits.write" in device:
                update_limit(device["limits.write"], read=0, write=1)


class StorageResourceCollector(StorageCollector):
    SPACE_TOTAL = Metric(
        namespace="lxd",
        subsystem="storage",
        name="space",
        unit="total",
        help="Storage pool total space",
        labelnames=("pool",),
    )

    SPACE_USED = Metric(
        namespace="lxd",
        subsystem="storage",
        name="space",
        unit="used",
        help="Storage pool used space",
        labelnames=("pool",),
    )

    INODES_TOTAL = Metric(
        namespace="lxd",
        subsystem="storage",
        name="inodes",
        unit="total",
        help="Storage pool total space",
        labelnames=("pool",),
    )

    INODES_USED = Metric(
        namespace="lxd",
        subsystem="storage",
        name="inodes",
        unit="used",
        help="Storage pool used space",
        labelnames=("pool",),
    )

    def update(self, storage: dict):
        resources: dict = storage["resources"]

        self.SPACE_TOTAL.labels(pool=storage["name"]).set(
            resources["space"]["total"],
        )
        self.SPACE_USED.labels(pool=storage["name"]).set(
            resources["space"]["used"],
        )
        self.INODES_TOTAL.labels(pool=storage["name"]).set(
            resources["inodes"]["total"],
        )
        self.INODES_USED.labels(pool=storage["name"]).set(
            resources["inodes"]["used"],
        )


class ProfileUsageCollector(ProfileCollector):
    METRIC_PROFILE = Metric(
        namespace="lxd",
        subsystem="profile",
        name="usage",
        unit="count",
        help="Profile metrics",
        labelnames=("profile",),
    )

    def update(self, profile: dict):
        self.METRIC_PROFILE.labels(
            profile=profile["name"],
        ).set(
            len(profile["used_by"]),
        )


ContainersTotal = Metric(
    namespace="lxd",
    subsystem="container",
    name="count",
    unit="",
    help="Total container count",
    labelnames=("status",),
)


CONTAINER_METRICS_REGISTRY: Mapping[str, ContainerCollector] = (
    MappingProxyType({
        "boot.autostart.priority": BootPriorityCollector(),
        "image.os": ImageOSCollector(),
        "image.release": ImageOSReleaseCollector(),
        "image.version": ImageOSVersionCollector(),
        "image.serial": ImageOSSerialCollector(),
        "limits.cpu": LimitsCPUCollector(),
        "limits.processes": LimitsProcessesCollector(),
        "limits.memory": LimitsMemoryCollector(),
    })
)


CONTAINER_VIRTUAL_METRICS_REGISTRY: Iterable[ContainerVirtualCollector] = (
    LimitsCPUEffectiveCollector(),
    StateCollector(),
)


CONTAINER_DEVICE_REGISTRY: Mapping[str, Iterable[ContainerDeviceCollector]] = (
    MappingProxyType({
        "disk": (ContainerDiskCollector(),),
    })
)

STORAGE_REGISTRY: Iterable[StorageCollector] = (
    StorageResourceCollector(),
)


PROFILES_REGISTRY: Iterable[ProfileCollector] = (
    ProfileUsageCollector(),
)


class CollectorService(PeriodicService):
    __required__ = (
        "lxd_url", "lxd_cert", "client_cert", "client_key",
    ) + tuple(PeriodicService.__required__)

    lxd_url: URL
    lxd_cert: Path
    client_cert: Path
    client_key: Path
    skip_interfaces: FrozenSet[str]

    _ssl_context: Optional[ssl.SSLContext]
    _client: aiohttp.ClientSession

    async def do_request(self, path: str, method="GET", **kwargs) -> dict:
        async with self._client.request(
            method=method,
            url=str(self.lxd_url).rstrip("/") + path,
            ssl_context=self._ssl_context,
            **kwargs
        ) as response:
            return await response.json()

    async def callback(self) -> Any:
        container_states = Counter(CONTAINER_STATUSES)

        instances = await self.do_request("/1.0/instances?recursion=2")

        container: dict
        for container in instances["metadata"]:
            container_states[container["status"].lower()] += 1

            for key, value in container["expanded_config"].items():
                if key not in CONTAINER_METRICS_REGISTRY:
                    continue

                collector_instance: ContainerCollector = (
                    CONTAINER_METRICS_REGISTRY[key]
                )
                try:
                    collector_instance.update(container, value)
                except Exception:
                    logging.exception("Failed to colelct metric %r", key)

            for collector_virtual in CONTAINER_VIRTUAL_METRICS_REGISTRY:
                try:
                    collector_virtual.update(container)
                except Exception:
                    logging.exception(
                        "Failed to update virtual metric collector %r",
                        collector_virtual,
                    )

            for name, device in container["expanded_devices"].items():
                device_type = device.get("type")
                for device_collector in CONTAINER_DEVICE_REGISTRY.get(
                    device_type, [],
                ):
                    try:
                        device_collector.update(container, name, device)
                    except Exception:
                        logging.exception(
                            "Failed to collect device %r with collector %r",
                            name, device_collector,
                        )

        for status, value in container_states.items():
            ContainersTotal.labels(status=status).set(value)

        storages = await self.do_request("/1.0/storage-pools?recursion=1")

        for storage in storages["metadata"]:

            resources = await self.do_request(
                f"/1.0/storage-pools/{storage['name']}/resources",
            )
            storage["resources"] = resources["metadata"]

            for storage_collector in STORAGE_REGISTRY:
                try:
                    storage_collector.update(storage)
                except Exception:
                    logging.exception(
                        "Failed to collect storage %r with collector %r",
                        storage, storage_collector,
                    )

        profiles = await self.do_request("/1.0/profiles?recursion=1")

        for profile in profiles["metadata"]:
            for profile_collector in PROFILES_REGISTRY:
                try:
                    profile_collector.update(profile)
                except Exception:
                    logging.exception(
                        "Failed to collect profile %r with collector %r",
                        profile.name, profile_collector,
                    )

    async def start(self):
        if self.lxd_url.scheme == "unix":
            path = Path(self.lxd_url.path)

            if not path.is_socket():
                raise RuntimeError("Path %s is not a unix socket" % path)
            connector = aiohttp.UnixConnector(path=str(path))
            self._ssl_context = None
            self.lxd_url = URL.build(
                scheme="http",
                host="lxd",
                user=self.lxd_url.user or "",
                password=self.lxd_url.password or "",
                query_string=self.lxd_url.query_string or "",
                authority=self.lxd_url.authority or "",
                port=self.lxd_url.port,
                path="/",
            )
        else:
            connector = aiohttp.TCPConnector()
            self._ssl_context = ssl.SSLContext()
            self._ssl_context.load_verify_locations(
                str(self.lxd_cert.expanduser()),
            )
            self._ssl_context.load_cert_chain(
                str(self.client_cert.expanduser()),
                str(self.client_key.expanduser()),
            )

        self._client = aiohttp.ClientSession(
            connector=connector,
            connector_owner=True,
            raise_for_status=True,
        )

        await super().start()


def main():
    arguments = Parser(
        auto_env_var_prefix="LXD_EXPORTER_",
        config_files=[
            "~/.config/lxd-exporter.ini",
            os.getenv("LXD_EXPORTER_CONFIG", "/etc/lxd-exporter.ini"),
        ],
    )
    arguments.parse_args()
    arguments.sanitize_env()

    basic_config(
        log_format=arguments.log.format,
        level=arguments.log.level,
    )

    if arguments.collector.skip_interface:
        logging.info(
            "Network interfaces starts with %r will be skipped",
            list(arguments.collector.skip_interface),
        )

    StateCollector.SKIP_INTERFACES = arguments.collector.skip_interface

    services = [
        MetricsAPI(
            address=arguments.http.address,
            port=arguments.http.port,
        ),
        CollectorService(
            interval=arguments.collector.interval,
            delay=arguments.collector.delay,
            lxd_url=arguments.lxd.url,
            lxd_cert=arguments.lxd.server_cert,
            client_cert=arguments.lxd.client_cert,
            client_key=arguments.lxd.client_key,
        ),
        SDWatchdogService(),
    ]

    with entrypoint(
        *services,
        log_format=arguments.log.format,
        log_level=arguments.log.level,
        pool_size=arguments.pool_size,
    ) as loop:
        loop.run_forever()


if __name__ == "__main__":
    main()
