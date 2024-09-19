import requests
import xml.etree.ElementTree as ET
from google.transit import gtfs_realtime_pb2
import time


def get_gtfs_feed(url, username, password):
    response = requests.get(url, auth=(username, password))
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(response.content)
    return feed


def write_line_to_csv(entity, line_id):
    timestamp = entity.timestamp
    start_time = entity.trip.start_time
    start_date = entity.trip.start_date
    lat = entity.position.latitude
    lon = entity.position.longitude
    bearing = entity.position.bearing
    speed = entity.position.speed
    direction = entity.trip.direction_id
    schedule_relationship = entity.trip.schedule_relationship
    current_stop_sequence = entity.current_stop_sequence
    stop_id = entity.stop_id
    current_status = entity.current_status

    csv_string = f"{timestamp},{start_time},{start_date},{lat},{lon},{bearing},{speed},{direction},{schedule_relationship},{current_stop_sequence},{stop_id},{current_status}"

    f = open(f"tampere_{line_id}.csv", "a")
    f.write(csv_string + "\n")
    f.close()


def cont_parse_gtfs(url, username, password):
    while True:
        try:
            feed = get_gtfs_feed(url, username, password)
        except Exception as e:
            print(e)
            continue
        for entity in feed.entity:
            if entity.id == "3012_MiCa":
                entity_301 = entity.vehicle
            if entity.id == "3012_Karsan":
                entity_303 = entity.vehicle
        write_line_to_csv(entity_301, "301")
        write_line_to_csv(entity_303, "303")
        time.sleep(1)


if __name__ == "__main__":
    # Setup the API params
    username = "8604566007446482"
    password = "4WMZWlgtZUfbF2FhOBurw57FoBEgGEm2"
    url = "https://data.waltti.fi/tampere/api/gtfsrealtime/v1.0/feed/vehicleposition"
    cont_parse_gtfs(url, username, password)
