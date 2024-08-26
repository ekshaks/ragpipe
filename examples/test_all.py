import pytest

#@pytest.mark.skip(reason="This test is disabled temporarily")

def test_startups():
    from .startups import Workflow
    docs = Workflow().run(respond_flag=False)
    assert len(docs) > 0

def test_insurance():
    from .insurance.insurance import Workflow
    docs = Workflow().run(respond_flag=False)
    assert len(docs) > 0

def test_billionaires():
    from .billionaires import Workflow
    docs = Workflow().run(respond_flag=False)
    assert len(docs) > 0
