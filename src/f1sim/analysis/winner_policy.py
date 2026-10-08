"""Choose the live reporting policy without changing recorded allocations."""


def simulation_winner_allocation(
    loader, year, race, drivers, weather, *, qualifying_weather=None,
    weather_schedule=None, starting_grid=None,
):
    """Retain native chances only in the adapter's verified practice context.

    Legacy/custom adapters keep their earlier three-argument allocation hook.
    Saved simulations continue to use the allocation stored with their inputs.
    """
    prefer_native = getattr(loader, "prefer_native_winner_forecast", None)
    if callable(prefer_native) and prefer_native(
        year, race, drivers, weather, qualifying_weather=qualifying_weather,
        weather_schedule=weather_schedule, starting_grid=starting_grid,
    ):
        return None
    allocation = getattr(loader, "get_winner_allocation", None)
    return allocation(year, race, drivers) if callable(allocation) else None
