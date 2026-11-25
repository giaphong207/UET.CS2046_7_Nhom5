"""Share one fitted synthetic A4 fixture across regression and V3 integrity tests."""
import pytest


@pytest.fixture(scope='session')
def clean_a4(tmp_path_factory):
    from diabetes_research import modeling
    from test_protocol import toy_data
    root=tmp_path_factory.mktemp('synthetic_a4')
    data=root/'synthetic.csv'
    toy_data().to_csv(data,index=False)
    output=root/'a4'
    manifest=modeling.run(data,output,trials=10,cv_folds=3,demo=True)
    return data,output,manifest
