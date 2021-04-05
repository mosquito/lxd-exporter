# lxd-exporter

Prometheus exporter for LXD

## Installation

Install from sources:

```bash
python3.8 -m venv /usr/share/python3/lxd-exporter
/usr/share/python3/lxd-exporter/bin/pip install -U pip wheel
/usr/share/python3/lxd-exporter/bin/pip install git+http://github.com/mosquito/lxd-exporter.git
ln -snf /usr/share/python3/lxd-exporter/bin/lxd-exporter /usr/local/bin
```

Run it:

```bash
APP_LISTEN=0.0.0.0 APP_PORT=8080 /usr/local/bin/lxd-exporter 
```

## Configuration

| Environment Variable | Default | Description |
| -- | -- | -- |
| `COLLECTOR_UPDATE_PERIOD` | 5 | How often collector will gather information from LXD daemon |
| `LXD_ENDPOINT_PASSWORD` | - | LXD daemon password, useful when access to LXD daemon via network |
| `LXD_ENDPOINT_CERT` | - | Client ssl cert , useful when access to LXD daemon via network |
| `LXD_ENDPOINT_KEY` | - | Client ssl key, useful when access to LXD daemon via network |
| `LXD_ENDPOINT` | - | LXD endpoint URL, useful when access to LXD daemon via network |
| `LXD_ENDPOINT_VERIFY_SSL` | 1 | Disables SSL certificate issuer, useful when access to LXD daemon via network |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `FATAL` |
| `APP_LISTEN` | `::1` | Service listen address |
| `APP_PORT` | `8080` | Service listen port |
