import pytest

#@pytest.mark.skip(reason="This test is disabled temporarily")

def test_startups():
    from .startups import main
    docs = main(respond_flag=False)
    assert len(docs) > 0

def test_insurance():
    from .insurance.insurance import main
    docs = main(respond_flag=False)
    assert len(docs) > 0

def test_billionaires():
    from .billionaires import main
    docs = main(respond_flag=False)
    assert len(docs) > 0
