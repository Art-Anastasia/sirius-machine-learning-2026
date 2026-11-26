"""Заранее заполнить локальный кэш Olivetti Faces в папке семинара."""

from pathlib import Path

from sklearn.datasets import fetch_olivetti_faces


DATA_HOME = Path(__file__).resolve().parent / "data_cache"


def main() -> None:
    faces = fetch_olivetti_faces(data_home=str(DATA_HOME), download_if_missing=True)
    assert faces.images.shape == (400, 64, 64)
    assert faces.target.shape == (400,)
    cached = fetch_olivetti_faces(data_home=str(DATA_HOME), download_if_missing=False)
    assert cached.images.shape == faces.images.shape
    print(f"Кэш готов: {DATA_HOME}")
    print("Проверка без загрузки из сети: 400 портретов, 40 человек.")


if __name__ == "__main__":
    main()
