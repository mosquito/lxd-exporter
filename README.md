# lxd-exporter

Prometheus exporter for LXD

## Installation

Install from sources:

```bash
python3.8 -m venv /usr/share/python3/lxd-exporter
/usr/share/python3/lxd-exporter/bin/pip install -U pip wheel
/usr/share/python3/lxd-exporter/bin/pip install lxd-exporter
ln -snf /usr/share/python3/lxd-exporter/bin/lxd-exporter /usr/local/bin
```

Run it:

```bash
/usr/local/bin/lxd-exporter 
```

## Configuration

### From command line

```
usage: lxd-exporter [-h] [--pool-size POOL_SIZE]
                    [--log-level {debug,info,warning,error,critical}]
                    [--log-format {stream,color,json,syslog,plain,journald,rich,rich_tb}]
                    [--lxd-url LXD_URL] [--lxd-server-cert LXD_SERVER_CERT]
                    [--lxd-client-cert LXD_CLIENT_CERT]
                    [--lxd-client-key LXD_CLIENT_KEY]
                    [--http-address HTTP_ADDRESS] [--http-port HTTP_PORT]
                    [--collector-interval COLLECTOR_INTERVAL]
                    [--collector-delay COLLECTOR_DELAY]
                    [--collector-skip-interface COLLECTOR_SKIP_INTERFACE [COLLECTOR_SKIP_INTERFACE ...]]

optional arguments:
  -h, --help            show this help message and exit
  --pool-size POOL_SIZE
                        (default: 4) [ENV: LXD_EXPORTER_POOL_SIZE]

Logging options:
  --log-level {debug,info,warning,error,critical}
                        (default: info) [ENV: LXD_EXPORTER_LOG_LEVEL]
  --log-format {stream,color,json,syslog,plain,journald,rich,rich_tb}
                        (default: color) [ENV: LXD_EXPORTER_LOG_FORMAT]

LXD options:
  --lxd-url LXD_URL     (default: unix:///var/snap/lxd/common/lxd/unix.socket)
                        [ENV: LXD_EXPORTER_LXD_URL]
  --lxd-server-cert LXD_SERVER_CERT
                        [ENV: LXD_EXPORTER_LXD_SERVER_CERT]
  --lxd-client-cert LXD_CLIENT_CERT
                        [ENV: LXD_EXPORTER_LXD_CLIENT_CERT]
  --lxd-client-key LXD_CLIENT_KEY
                        [ENV: LXD_EXPORTER_LXD_CLIENT_KEY]

HTTP server options:
  --http-address HTTP_ADDRESS
                        (default: 127.0.0.1) [ENV: LXD_EXPORTER_HTTP_ADDRESS]
  --http-port HTTP_PORT
                        (default: 8080) [ENV: LXD_EXPORTER_HTTP_PORT]

Collector Service options:
  --collector-interval COLLECTOR_INTERVAL
                        (default: 30) [ENV: LXD_EXPORTER_COLLECTOR_INTERVAL]
  --collector-delay COLLECTOR_DELAY
                        (default: 0) [ENV: LXD_EXPORTER_COLLECTOR_DELAY]
  --collector-skip-interface COLLECTOR_SKIP_INTERFACE [COLLECTOR_SKIP_INTERFACE ...]
                        (default: []) [ENV:
                        LXD_EXPORTER_COLLECTOR_SKIP_INTERFACE]

Default values will based on following configuration files ['~/.config/lxd-
exporter.ini', '/etc/lxd-exporter.ini']. Now 1 files has been applied
['/Users/mosquito/.config/lxd-exporter.ini']. The configuration files is INI-
formatted files where configuration groups is INI sections.See more
https://pypi.org/project/argclass/#configs
```

### From config file

Example config file:

```ini
[DEFAULT]
pool_size = 4

[http]
address = 0.0.0.0
port = 8123

[lxd]
url = https://lxd.example.net:8443
server_cert = ~/.config/lxc/servercerts/example.crt
client_key = ~/.config/lxc/client.key
client_cert = ~/.config/lxc/client.crt

[collector]
delay = 1
interval = 15
skip_interface = ["docker", "lo"]

[log]
level = info
format = stream
```

### From environment

| Environment Variable | Default | Description |
| -- | -- | -- |
| `LXD_EXPORTER_CONFIG` | `/etc/lxd-exporter.ini` | Read the configuration from this config file if exists | 
| `LXD_EXPORTER_COLLECTOR_DELAY` | `0` | Delay before collector starts gathering info |
| `LXD_EXPORTER_COLLECTOR_INTERVAL` | `30` | How often collector will gather information from LXD daemon |
| `LXD_EXPORTER_COLLECTOR_SKIP_INTERFACE` | `[]` | List of skipping interface prefixes |
| `LXD_EXPORTER_HTTP_ADDRESS` | `127.0.0.1` | Service listen address |
| `LXD_EXPORTER_HTTP_PORT` | `8080` | Service listen port |
| `LXD_EXPORTER_LOG_LEVEL` | `color` (`journald` if available) | Logging output format (`stream`, `color`, `json`, `syslog`, `plain`, `journald`, `rich`, `rich_tb`) |
| `LXD_EXPORTER_LOG_LEVEL` | `info` | Logging level `debug`, `info`, `warning`, `error`, `fatal` |
| `LXD_EXPORTER_LXD_CLIENT_CERT` | - | Path to LXD ssl client certificate |
| `LXD_EXPORTER_LXD_CLIENT_KEY` | - | Path to LXD ssl client key |
| `LXD_EXPORTER_LXD_SERVER_CERT` | - | Path to LXD server ssl certificate |
| `LXD_EXPORTER_LXD_URL` | `unix:///var/snap/lxd/common/lxd/unix.socket` | LXD endpoint URL, useful when access to LXD daemon via network |
| `LXD_EXPORTER_POOL_SIZE` | `4` | Thread pool size |
