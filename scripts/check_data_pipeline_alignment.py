
# Import path configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查数据链路中不同位置的数据集是否对齐：
1. data_w2ner_folded/
2. data_w2ner/
3. W2NER/data/data_w2ner_folded_with_dev/
4. W2NER/data/w2ner_format/
"""

import json
import hashlib
from pathlib import Path

# 数据目录
DATA_LOCATIONS = {
    "data_w2ner_folded": project_root / "data_w2ner_folded",
    "data_w2ner": project_root / "data_w2ner",
    "W2NER/data/data_w2ner_folded_with_dev": project_root / "W2NER" / "data" / "data_w2ner_folded_with_dev",
    "W2NER/data/w2ner_format": project_root / "W2NER" / "data" / "w2ner_format",
}

DATASETS = [
    "flight_orders_with_queries",
    "hotel_orders_with_queries",
    "id_cards_with_queries",
    "mixed_data_with_queries",
    "train_orders_with_queries",
]

SPLITS = ["train", "dev", "test"]


def md5_file(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=" * 90)
    print(" " * 20 + "数据链路对齐检查报告")
    print("=" * 90)
    
    for dataset in DATASETS:
        print(f"\n{'='*90}")
        print(f"【{dataset}】")
        print("=" * 90)
        
        for split in SPLITS:
            print(f"\n  --- {split}.json ---")
            
            md5_map = {}
            sample_counts = {}
            
            for loc_name, loc_path in DATA_LOCATIONS.items():
                file_path = loc_path / dataset / f"{split}.json"
                
                if file_path.exists():
                    md5 = md5_file(file_path)
                    md5_map[loc_name] = md5
                    
                    data = load_json(file_path)
                    sample_counts[loc_name] = len(data)
                else:
                    md5_map[loc_name] = "N/A"
                    sample_counts[loc_name] = "N/A"
            
            # 打印 MD5
            print(f"  {'位置':<45} {'MD5':<35} {'样本数':>10}")
            print(f"  {'-'*45} {'-'*35} {'-'*10}")
            
            for loc_name in md5_map:
                md5 = md5_map[loc_name]
                count = sample_counts[loc_name]
                md5_short = md5[:32] if md5 != "N/A" else "N/A"
                print(f"  {loc_name:<45} {md5_short:<35} {count:>10}")
            
            # 检查是否一致
            unique_md5s = set(v for v in md5_map.values() if v != "N/A")
            if len(unique_md5s) == 1:
                print(f"  ✅ 所有位置数据一致")
            elif len(unique_md5s) > 1:
                print(f"  ❌ 数据不一致！发现 {len(unique_md5s)} 种不同版本")
                
                # 分组显示
                md5_groups = {}
                for loc_name, md5 in md5_map.items():
                    if md5 != "N/A":
                        if md5 not in md5_groups:
                            md5_groups[md5] = []
                        md5_groups[md5].append(loc_name)
                
                for md5, locs in md5_groups.items():
                    print(f"     版本 {md5[:16]}...: {', '.join(locs)}")
    
    print("\n" + "=" * 90)
    print("检查完成")
    print("=" * 90)


if __name__ == "__main__":
    main()
