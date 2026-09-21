"""Strict validation for model values stored in saved JSON inputs."""

import json


def validate_saved_model(model_type, value):
    """Validate a decoded saved model with the replay JSON contract.

    Re-serializing the decoded value preserves JSON enum strings and arrays used
    for tuple fields while strict Pydantic JSON validation rejects booleans and
    quoted numbers in numeric fields.
    """
    return model_type.model_validate_json(json.dumps(value), strict=True)
