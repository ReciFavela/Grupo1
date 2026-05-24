from influxdb_client import InfluxDBClient, Point

from src.config import (
    INFLUX_URL,
    INFLUX_TOKEN,
    INFLUX_ORG,
    INFLUX_BUCKET
)

client = InfluxDBClient(
    url=INFLUX_URL,
    token=INFLUX_TOKEN,
    org=INFLUX_ORG
)

write_api = client.write_api()


def send_metrics(result):

    point = (
        Point("detections")
        .tag("class", result["class"])
        .field("confidence", float(result["confidence"]))
    )

    write_api.write(
        bucket=INFLUX_BUCKET,
        org=INFLUX_ORG,
        record=point
    )