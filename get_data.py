from FRED_Loader.load import pull_fred
from FRED_Loader.series import ALL_SERIES
from FRED_Loader.utils import Config

FILENAME: str = "fred_data.csv"
OUTPUT_PATH: str = "data/"
START_DATE: str = "2000-01-01"
RESAMPLE_RULE: str = "W-FRI"
MEAN_FREQS: set[str] = {"D"}
APPLY_SCORES: bool = True
SERIES_OF_INTEREST = {**ALL_SERIES}


def main():

    cfg: Config = Config(
        filename=FILENAME,
        output_path=OUTPUT_PATH,
        start=START_DATE,
        resample_rule=RESAMPLE_RULE,
        mean_freqs=MEAN_FREQS,
        series=SERIES_OF_INTEREST,
    )

    _ = pull_fred(config=cfg, apply_scores=APPLY_SCORES)


if __name__ == "__main__":
    main()
