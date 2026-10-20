import pandas as pd
from src.data import FEATURES, TARGET, load_dataset


def test_loader_converts_missing_and_binarizes(tmp_path):
    rows = (
        "63,1,3,145,233,1,0,150,0,2.3,3,0,6,0\n67,1,4,160,286,0,2,108,1,1.5,2,3,?,4\n"
    )
    path = tmp_path / "data.csv"
    path.write_text(rows, encoding="utf-8")
    frame = load_dataset(path)
    assert list(frame.columns) == FEATURES + [TARGET]
    assert frame[TARGET].tolist() == [0, 1]
    assert pd.isna(frame.loc[1, "thal"])
