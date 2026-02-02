#!/usr/bin/env python3
"""
Voyager Windows 診斷腳本
========================
用於診斷 Voyager 在 Windows 上的 save/load 問題。

執行方式:
    python scripts/debug_voyager_windows.py
"""
import sys
import tempfile
import platform
from pathlib import Path

import numpy as np

def main():
    print("=" * 60)
    print("Voyager Windows 診斷")
    print("=" * 60)

    # 環境資訊
    print(f"\n[環境資訊]")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print(f"  Python: {sys.version}")

    try:
        import voyager
        print(f"  Voyager: 已安裝")
        print(f"  Voyager 屬性: {[x for x in dir(voyager) if not x.startswith('_')]}")
    except ImportError as e:
        print(f"  Voyager: 未安裝 ({e})")
        return 1

    try:
        import importlib.metadata
        version = importlib.metadata.version('voyager')
        print(f"  Voyager 版本: {version}")
    except Exception:
        print(f"  Voyager 版本: 無法取得")

    # 測試 1: 基本建立和搜尋
    print(f"\n[測試 1] 基本建立和搜尋")
    try:
        np.random.seed(42)
        dim = 128
        num_vectors = 100
        vectors = np.random.randn(num_vectors, dim).astype(np.float32)

        index = voyager.Index(
            space=voyager.Space.Cosine,
            num_dimensions=dim,
            M=12,
            ef_construction=200,
            storage_data_type=voyager.StorageDataType.Float32,
        )
        index.add_items(vectors)

        # 搜尋測試
        query = vectors[0:1]
        neighbors, distances = index.query(query, k=5)
        print(f"  ✓ 建立成功: {len(index)} vectors")
        print(f"  ✓ 搜尋成功: neighbors={neighbors[0].tolist()}, distances={distances[0].tolist()[:3]}...")
    except Exception as e:
        print(f"  ✗ 失敗: {e}")
        return 1

    # 測試 2: Save 到檔案
    print(f"\n[測試 2] Save 到檔案")
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_index.voyager"

        try:
            index.save(str(save_path))
            file_size = save_path.stat().st_size
            print(f"  ✓ Save 成功: {save_path}")
            print(f"  ✓ 檔案大小: {file_size} bytes")
        except Exception as e:
            print(f"  ✗ Save 失敗: {e}")
            return 1

        # 測試 3a: Load (無參數 - 現代 API)
        print(f"\n[測試 3a] Load (無參數 - 現代 Voyager API)")
        try:
            index2 = voyager.Index.load(str(save_path))
            print(f"  ✓ Load 成功 (無參數): {len(index2)} vectors")

            # 驗證搜尋
            neighbors2, distances2 = index2.query(query, k=5)
            print(f"  ✓ 搜尋驗證: neighbors={neighbors2[0].tolist()}")
        except Exception as e:
            print(f"  ✗ Load 失敗 (無參數): {e}")
            print(f"    → 這表示 Voyager 在此平台可能有 save/load 相容性問題")

        # 測試 3b: Load (有參數 - 舊版 API)
        print(f"\n[測試 3b] Load (有參數 - 舊版 Voyager API)")
        try:
            index3 = voyager.Index.load(
                str(save_path),
                space=voyager.Space.Cosine,
                num_dimensions=dim,
                storage_data_type=voyager.StorageDataType.Float32,
            )
            print(f"  ✓ Load 成功 (有參數): {len(index3)} vectors")

            # 驗證搜尋
            neighbors3, distances3 = index3.query(query, k=5)
            print(f"  ✓ 搜尋驗證: neighbors={neighbors3[0].tolist()}")
        except Exception as e:
            print(f"  ✗ Load 失敗 (有參數): {e}")

        # 測試 3c: Load 使用 binary file handle
        print(f"\n[測試 3c] Load (使用 file handle)")
        try:
            with open(save_path, 'rb') as f:
                index4 = voyager.Index.load(f)
            print(f"  ✓ Load 成功 (file handle): {len(index4)} vectors")
        except Exception as e:
            print(f"  ✗ Load 失敗 (file handle): {e}")

        # 測試 3d: Load 使用 file handle + 參數
        print(f"\n[測試 3d] Load (file handle + 參數)")
        try:
            with open(save_path, 'rb') as f:
                index5 = voyager.Index.load(
                    f,
                    space=voyager.Space.Cosine,
                    num_dimensions=dim,
                    storage_data_type=voyager.StorageDataType.Float32,
                )
            print(f"  ✓ Load 成功 (file handle + 參數): {len(index5)} vectors")
        except Exception as e:
            print(f"  ✗ Load 失敗 (file handle + 參數): {e}")

        # 測試 4: 檢查檔案內容
        print(f"\n[測試 4] 檔案內容檢查")
        try:
            with open(save_path, 'rb') as f:
                header = f.read(64)
            print(f"  檔案前 64 bytes (hex): {header[:32].hex()}")
            print(f"  檔案前 64 bytes (ascii): {header[:32]}")
        except Exception as e:
            print(f"  ✗ 讀取失敗: {e}")

    print("\n" + "=" * 60)
    print("診斷完成")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
