from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from case_web.inference import UPLOADS_DIR, ensure_runtime_dirs, run_test_inference


def extract_npz_from_zip(zip_path: Path, password: str) -> Path:
    ensure_runtime_dirs()
    with zipfile.ZipFile(zip_path) as archive:
        npz_names = [name for name in archive.namelist() if name.lower().endswith(".npz")]
        if not npz_names:
            raise FileNotFoundError("В архиве не найден .npz файл")
        npz_name = npz_names[0]
        output_path = UPLOADS_DIR / Path(npz_name).name
        with archive.open(npz_name, pwd=password.encode("utf-8")) as src, output_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск инференса на парольном тестовом архиве")
    parser.add_argument("--zip-path", required=True, help="Путь до архива Answers_reduced.zip")
    parser.add_argument("--password", required=True, help="Пароль архива от жюри")
    args = parser.parse_args()

    zip_path = Path(args.zip_path).expanduser().resolve()
    npz_path = extract_npz_from_zip(zip_path, args.password)
    result = run_test_inference(npz_path)

    print("Файл обработан:")
    print(f"  npz: {npz_path}")
    print(f"  samples: {result['num_samples']}")
    print(f"  accuracy: {result['final_accuracy']}")
    print(f"  loss: {result['final_loss']}")


if __name__ == "__main__":
    main()
