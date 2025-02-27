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


def test_sec10k():
    from .sec10k.sec10k import Workflow
    response = input("Run sec10k test (needs GPU support)? (y/n): ")
    if response.lower() != 'y':
        pytest.exit("Tests not run. Run separately as `python -m examples.sec10k.sec10k`")
    docs = Workflow().run(respond_flag=False)
    assert len(docs) > 0

'''
def test_billionaires():
    from .billionaires import Workflow
    docs = Workflow().run(respond_flag=False)
    assert len(docs) > 0
'''

