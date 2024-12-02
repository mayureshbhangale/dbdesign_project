from ensemble_compilation.graph_representation import SchemaGraph, Table

def gen_flights_1B_schema(csv_path):
    schema = SchemaGraph()

    attributes = [
        'YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'FLIGHT_NUMBER',
        'TAIL_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
        'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT',
        'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE',
        'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME',
        'ARRIVAL_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON',
        'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY',
        'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'
    ]

    primary_key = ['YEAR', 'MONTH', 'DAY', 'FLIGHT_NUMBER', 'TAIL_NUMBER']

    flights = Table(
        table_name='flights',
        attributes=attributes,
        csv_file_location=f'{csv_path}',
        table_size=5819080,  # Adjust if needed
        primary_key=primary_key,
        sample_rate=1.0
    )

    schema.add_table(flights)

    return schema
