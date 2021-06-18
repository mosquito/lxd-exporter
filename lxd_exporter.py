from collections import Counter

import gevent

from abc import ABC, abstractmethod
from types import MappingProxyType
import re
from typing import Mapping, Iterable, Any, Union

import logging
import os

from flask import Flask
from pylxd import Client
from pylxd.models import Container, StoragePool, StorageResources, Profile
from pylxd.models.instance import InstanceState
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from prometheus_client import make_wsgi_app, Gauge, Counter as MetricCounter
from prometheus_client.utils import INF
from gevent.pywsgi import WSGIServer
from gevent import sleep


app = Flask(__name__)
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

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

UPDATE_PERIOD = int(os.getenv('COLLECTOR_UPDATE_PERIOD', '5'))
INTERFACE_SKIPS = tuple(os.getenv('INTERFACE_SKIPS', '').split(","))


def is_interface_skipped(name):
    for skip in INTERFACE_SKIPS:
        if name.startswith(skip):
            return True
    return False


def make_client():
    lxd_password = os.getenv("LXD_ENDPOINT_PASSWORD")

    lxd_cert = os.getenv("LXD_ENDPOINT_CERT")
    lxd_key = os.getenv("LXD_ENDPOINT_KEY")

    client = Client(
        endpoint=os.getenv("LXD_ENDPOINT"),
        verify=bool(int(os.getenv("LXD_ENDPOINT_VERIFY_SSL", '1'))),
        cert=(lxd_cert, lxd_key) if lxd_key is not None else None
    )

    if lxd_password:
        client.authenticate(lxd_password)

    return client


CLIENT = make_client()


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
            "ns": 0.000000001
        }
        value, suffix = re.match(r"^([\d.]+)(\w+)?$", human_value).groups()
        return float(value) * suffixes[suffix]

    @abstractmethod
    def update(self, *args):
        pass


class ContainerCollector(CollectorBase):
    @abstractmethod
    def update(self, container: Container, value: str):
        pass


class ContainerVirtualCollector(CollectorBase):
    @abstractmethod
    def update(self, container: Container):
        pass


class ContainerDeviceCollector(CollectorBase):
    @abstractmethod
    def update(self, container: Container, device_name: str, device: Mapping[str, Any]):
        pass


class StorageCollector(CollectorBase):
    @abstractmethod
    def update(self, storage: StoragePool):
        pass


class ProfileCollector(CollectorBase):
    @abstractmethod
    def update(self, profile: Profile):
        pass


def simple_collector(name, documentation, labels=("container", "location")):
    class SimpleCollector(ContainerCollector):
        METRIC = Gauge(
            name=name,
            documentation=documentation,
            labelnames=labels
        )

        def update(self, container: Container, value: str):
            self.METRIC.labels(
                container=container.name, location=container.location
            ).set(int(value))

    return SimpleCollector


BootPriorityCollector = simple_collector(
    name='lxd_container_boot_autostart_priority',
    documentation='Container boot autostart priority'
)


class ImageOSCollector(ContainerCollector):
    METRIC = Gauge(
        name="lxd_container_image_os",
        documentation="Container os name image string",
        labelnames=("container", "location", "os")
    )

    def update(self, container: Container, value: str):
        self.METRIC.labels(
            container=container.name, location=container.location, os=value
        ).set(1)


class ImageOSReleaseCollector(ContainerCollector):
    METRIC = Gauge(
        name="lxd_container_image_os_release",
        documentation="Container os release image string",
        labelnames=("container", "location", "release")
    )

    def update(self, container: Container, value: str):
        self.METRIC.labels(
            container=container.name, location=container.location, release=value
        ).set(1)


class ImageOSVersionCollector(ContainerCollector):
    METRIC = Gauge(
        name="lxd_container_image_os_version",
        documentation="Container os version image string",
        labelnames=("container", "location", "version")
    )

    def update(self, container: Container, value: str):
        self.METRIC.labels(
            container=container.name, location=container.location, version=value
        ).set(1)


class ImageOSSerialCollector(ContainerCollector):
    METRIC = Gauge(
        name="lxd_container_image_os_serial",
        documentation="Container os serial image string",
        labelnames=("container", "location", "serial")
    )

    def update(self, container: Container, value: str):
        self.METRIC.labels(
            container=container.name, location=container.location, serial=value
        ).set(1)


LimitsCPUCollector = simple_collector(
    name="lxd_container_limits_cpu",
    documentation="Container cpu limit"
)

LimitsProcessesCollector = simple_collector(
    name="lxd_container_limits_processes",
    documentation="Container processes limit",
)


class LimitsMemoryCollector(ContainerCollector):
    METRIC = Gauge(
        name="lxd_container_limits_memory",
        documentation="Container limits memory",
        labelnames=("container", "location")
    )

    def update(self, container: Container, value: str):
        self.METRIC.labels(
            container=container.name, location=container.location
        ).set(self.dehumanize_size(value))


class LimitsCPUEffectiveCollector(ContainerVirtualCollector):
    METRIC = Gauge(
        name="lxd_container_limits_cpu_effective",
        documentation="Container effective cpu limit",
        labelnames=("container", "location")
    )

    def get_cpu_limit(self, container: Container) -> Union[int, float]:
        cpus = container.expanded_config.get("limits.cpu")
        allowance = container.expanded_config.get("limits.cpu.allowance")

        if cpus is None:
            return INF

        value = int(cpus)
        if allowance is not None:
            allowed, period = map(self.dehumanize_time, allowance.split("/", 1))
            return allowed / period

        return value

    def update(self, container: Container):
        value = self.get_cpu_limit(container)

        self.METRIC.labels(
            container=container.name, location=container.location
        ).set(value)


class StateCollector(ContainerVirtualCollector):
    METRIC_CPU = Gauge(
        name="lxd_container_state_cpu",
        documentation="Container CPU state",
        labelnames=("container", "location")
    )

    METRIC_PROCESSES = Gauge(
        name="lxd_container_state_processes",
        documentation="Container running processes",
        labelnames=("container", "location")
    )

    METRIC_DISK = Gauge(
        name="lxd_container_state_disk_usage",
        documentation="Container disk device statistic",
        labelnames=("container", "location", "device", "pool", "path")
    )

    METRIC_MEMORY = Gauge(
        name="lxd_container_state_memory_usage",
        documentation="Container memory usage",
        labelnames=("container", "location")
    )

    METRIC_MEMORY_PEAK = Gauge(
        name="lxd_container_state_memory_usage_peak",
        documentation="Container memory peak usage",
        labelnames=("container", "location")
    )

    METRIC_SWAP = Gauge(
        name="lxd_container_state_swap_usage",
        documentation="Container swap usage",
        labelnames=("container", "location")
    )

    METRIC_SWAP_PEAK = Gauge(
        name="lxd_container_state_swap_usage_peak",
        documentation="Container swap peak usage",
        labelnames=("container", "location")
    )

    METRIC_IF_STATE = Gauge(
        name="lxd_container_state_network_interface",
        documentation="Container interface state",
        labelnames=("container", "location", "device", "state")
    )

    METRIC_IP = Gauge(
        name="lxd_container_state_network_addresses",
        documentation="Container IP addresses",
        labelnames=("container", "location", "device", "family")
    )

    METRIC_NETWORK_RX = Gauge(
        name="lxd_container_state_network_bytes_rx",
        documentation="Container network received bytes",
        labelnames=("container", "location", "device")
    )

    METRIC_NETWORK_TX = Gauge(
        name="lxd_container_state_network_bytes_tx",
        documentation="Container network transmitted bytes",
        labelnames=("container", "location", "device")
    )

    METRIC_NETWORK_PACKETS_RX = Gauge(
        name="lxd_container_state_network_packets_rx",
        documentation="Container network received packets",
        labelnames=("container", "location", "device")
    )

    METRIC_NETWORK_PACKETS_TX = Gauge(
        name="lxd_container_state_network_packets_tx",
        documentation="Container network transmitted bytes",
        labelnames=("container", "location", "device")
    )

    def update(self, container: Container):
        state: InstanceState = container.state()
        disks = {
            key: value for key, value in container.expanded_devices.items()
            if value.get('type') == 'disk'
        }
        labels = MappingProxyType(dict(container=container.name, location=container.location))

        self.METRIC_CPU.labels(**labels).set(state.cpu['usage'])
        self.METRIC_PROCESSES.labels(**labels).set(state.processes)

        for name, usage in state.disk.items():
            pool = disks.get(name, {}).get("pool")
            path = disks.get(name, {}).get("path")

            self.METRIC_DISK.labels(
                device=name, pool=pool, path=path, **labels
            ).set(usage['usage'])

        self.METRIC_MEMORY.labels(**labels).set(state.memory['usage'])
        self.METRIC_MEMORY_PEAK.labels(**labels).set(state.memory['usage_peak'])

        self.METRIC_SWAP.labels(**labels).set(state.memory['swap_usage'])
        self.METRIC_SWAP_PEAK.labels(**labels).set(state.memory['swap_usage_peak'])

        if not state.network:
            return

        for name, usage in state.network.items():
            if is_interface_skipped(name):
                continue

            self.METRIC_NETWORK_RX.labels(
                device=name, **labels
            ).set(usage['counters']['bytes_received'])
            self.METRIC_NETWORK_TX.labels(
                device=name, **labels
            ).set(usage['counters']['bytes_sent'])
            self.METRIC_NETWORK_PACKETS_RX.labels(
                device=name, **labels
            ).set(usage['counters']['packets_received'])
            self.METRIC_NETWORK_PACKETS_TX.labels(
                device=name, **labels
            ).set(usage['counters']['packets_sent'])
            self.METRIC_IF_STATE.labels(
                device=name, state=usage['state'], **labels
            ).set(1)

            ip_state = Counter()
            for address in usage['addresses']:
                ip_state[address['family']] += 1

            for family, value in ip_state.items():
                self.METRIC_IP.labels(
                    device=name, family=family, **labels
                ).set(value)


class ContainerDiskCollector(ContainerDeviceCollector):
    METRIC = Gauge(
        name="lxd_container_device_disk",
        documentation="Container disk device statistic",
        labelnames=("container", "location", "device", "pool", "path")
    )

    METRIC_SIZE = Gauge(
        name="lxd_container_device_disk_size",
        documentation="Container disk size device statistic",
        labelnames=("container", "location", "device", "pool", "path")
    )

    METRIC_LIMITS = Gauge(
        name="lxd_container_device_disk_limits",
        documentation="Container disk size device statistic",
        labelnames=("container", "location", "device", "pool", "path", "read", "write")
    )

    METRIC_LIMITS_IOPS = Gauge(
        name="lxd_container_device_disk_limits_iops",
        documentation="Container disk size device statistic",
        labelnames=("container", "location", "device", "pool", "path", "read", "write")
    )

    def update(self, container: Container, device_name: str, device: Mapping[str, Any]):
        labels = dict(container=container.name, location=container.location, device=device_name,
                      pool=device.get('pool'), path=device.get('path'))

        self.METRIC.labels(**labels).set(1)

        if 'size' in device:
            self.METRIC_SIZE.labels(**labels).set(self.dehumanize_size(device['size']))

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
            if 'limits.max' in device:
                update_limit(device['limits.max'], read=1, write=1)
            if 'limits.read' in device:
                update_limit(device['limits.read'], read=1, write=0)
            if 'limits.write' in device:
                update_limit(device['limits.write'], read=0, write=1)


class StorageResourceCollector(StorageCollector):
    METRIC_TOTAL = Gauge(
        name="lxd_storage_space_total",
        documentation="Storage pool total space",
        labelnames=("pool",)
    )

    METRIC_USED = Gauge(
        name="lxd_storage_space_used",
        documentation="Storage pool used space",
        labelnames=("pool",)
    )

    def update(self, storage: StoragePool):
        resources: StorageResources = storage.resources.get()

        self.METRIC_TOTAL.labels(pool=storage.name).set(resources.space['total'])
        self.METRIC_USED.labels(pool=storage.name).set(resources.space['used'])


class ProfileUsageCollector(ProfileCollector):
    METRIC_PROFILE = Gauge(
        name="lxd_profile_usage_count",
        documentation="Profile metrics",
        labelnames=('profile',)
    )

    def update(self, profile: Profile):
        self.METRIC_PROFILE.labels(profile=profile.name).set(len(profile.used_by))


ContainersTotal = Gauge(
    "lxd_container_count", "Total container count", labelnames=("status",)
)

UpdateTime = MetricCounter("lxd_exporter_heartbeat", documentation="Last update since exporter start")


CONTAINER_METRICS_REGISTRY: Mapping[str, ContainerCollector] = MappingProxyType({
    "boot.autostart.priority": BootPriorityCollector(),
    "image.os": ImageOSCollector(),
    "image.release": ImageOSReleaseCollector(),
    "image.version": ImageOSVersionCollector(),
    "image.serial": ImageOSSerialCollector(),
    "limits.cpu": LimitsCPUCollector(),
    "limits.processes": LimitsProcessesCollector(),
    "limits.memory": LimitsMemoryCollector()
})


CONTAINER_VIRTUAL_METRICS_REGISTRY: Iterable[ContainerVirtualCollector] = (
    LimitsCPUEffectiveCollector(),
    StateCollector(),
)


CONTAINER_DEVICE_REGISTRY: Mapping[str, Iterable[ContainerDeviceCollector]] = MappingProxyType({
    "disk": (ContainerDiskCollector(), )
})

STORAGE_REGISTRY: Iterable[StorageCollector] = (
    StorageResourceCollector(),
)


PROFILES_REGISTRY: Iterable[ProfileCollector] = (
    ProfileUsageCollector(),
)


def collect():
    containers = Counter(CONTAINER_STATUSES)

    for container in CLIENT.containers.all():
        containers[container.status.lower()] += 1

        for key, value in container.expanded_config.items():
            if key not in CONTAINER_METRICS_REGISTRY:
                logging.debug('Unhandled metric: %s = %r', key, value)
                continue

            collector_instance: ContainerCollector = CONTAINER_METRICS_REGISTRY[key]
            try:
                collector_instance.update(container, value)
            except Exception:
                logging.exception("Failed to colelct metric %r", key)

        for collector_virtual in CONTAINER_VIRTUAL_METRICS_REGISTRY:
            try:
                collector_virtual.update(container)
            except Exception:
                logging.exception("Failed to update virtual metric collector %r", collector_virtual)

        for name, device in container.expanded_devices.items():
            device_type = device.get('type')
            for device_collector in CONTAINER_DEVICE_REGISTRY.get(device_type, []):
                try:
                    device_collector.update(container, name, device)
                except Exception:
                    logging.exception("Failed to collect device %r with collector %r", name, device_collector)

    for status, value in containers.items():
        ContainersTotal.labels(status=status).set(value)

    for storage in CLIENT.storage_pools.all():
        for storage_collector in STORAGE_REGISTRY:
            try:
                storage_collector.update(storage)
            except Exception:
                logging.exception("Failed to collect device %r with collector %r", name, device_collector)

    for profile in CLIENT.profiles.all():
        for profile_collector in PROFILES_REGISTRY:
            try:
                profile_collector.update(profile)
            except Exception:
                logging.exception("Failed to collect profile %r with collector %r", profile.name, profile_collector)

    UpdateTime.inc(1)


def collector():
    while True:
        try:
            collect()
        except Exception:
            logging.exception("Error when collecting")
        finally:
            sleep(UPDATE_PERIOD)


collector_greenlet = gevent.greenlet.Greenlet(collector)
collector_greenlet.start()


def main():
    logging.basicConfig(
        level=getattr(
            logging, os.getenv("LOG_LEVEL", "INFO").upper(),
            logging.INFO
        )
    )
    address = os.getenv("APP_LISTEN", "::1")
    port = int(os.getenv('APP_PORT', '8080'))
    http_server = WSGIServer((address, port), app)
    logging.info("Listening %s:%d", address, port)
    http_server.serve_forever()


if __name__ == '__main__':
    main()
