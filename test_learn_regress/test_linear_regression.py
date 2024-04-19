from learn_regress import linear_regression

def test_haversine():
    # Amsterdam to Berlin
    assert linear_regression.haversine(
        4.895168, 52.370216, 13.404954, 52.520008
    ) == 576.6625818456291