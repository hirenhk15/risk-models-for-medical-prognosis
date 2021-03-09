# Tests for energy/data endpoint

import json
import pytest
from app import app


def test_endpoints(client):
    """
    Test function to test training and prediction
    endpoints and its response.
    """
    with open('./tests/data/test.json') as f:
        _param = json.load(f)

    # Test training endpoint
    res_train = client.get(
        '/train', content_type='application/json'
        )
    assert res_train.status_code == 200, (res_train.data, 'Training API failed!')
    
    # Test prediction endpoint
    res_predict = client.post(
        '/predict', data=json.dumps(_param), content_type='application/json'
        )
    assert res_predict.status_code == 200, (res_predict.data, 'Prediction API failed!')