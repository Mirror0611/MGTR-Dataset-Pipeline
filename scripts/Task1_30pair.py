import pandas as pd
import numpy as np
from datetime import timedelta
import os
from tqdm import tqdm

# ================= 配置区 =================
BASE_PATH = r"D:\科研\电子病历关系与时间建模\datasets\MIMIC-III-clinical-database\mimic-iii-clinical-database-1.4"
# SAMPLE_LIMIT = 5000
SAMPLE_LIMIT = 58976
# OUTPUT_FILE = "task1_30pair_5000.csv"
OUTPUT_FILE = "task1.csv"

# 精细化药理图谱
# pos_t: 判定有效的最小变化百分比
# stable_t: 判定为无显著变化的界限百分比
EVIDENCE_MAP = {
    # 1. 代谢与内分泌
    'Insulin': {'itemids': [50811, 50931, 51478], 'range': (20, 800), 'direction': -1, 'pos_t': 18.0, 'stable_t': 5.0,
                'name': 'Glucose'},
    'Potassium Chloride': {'itemids': [50971], 'range': (1.5, 9.0), 'direction': 1, 'pos_t': 10.0, 'stable_t': 3.0,
                           'name': 'Potassium'},
    'Sodium Bicarbonate': {'itemids': [50803, 50882], 'range': (5, 50), 'direction': 1, 'pos_t': 12.0, 'stable_t': 4.0,
                           'name': 'Bicarbonate'},
    'Levothyroxine': {'itemids': [50993], 'range': (0.01, 100), 'direction': -1, 'pos_t': 20.0, 'stable_t': 5.0,
                      'name': 'TSH'},
    'Prednisone': {'itemids': [50931, 50811], 'range': (20, 800), 'direction': 1, 'pos_t': 15.0, 'stable_t': 5.0,
                   'name': 'Glucose'},
    'Metformin': {'itemids': [50813], 'range': (0.1, 20.0), 'direction': 1, 'pos_t': 15.0, 'stable_t': 5.0,
                  'name': 'Lactate'},

    # 2. 肾脏与电解质
    'Furosemide': {'itemids': [50971], 'range': (1.5, 9.0), 'direction': -1, 'pos_t': 10.0, 'stable_t': 3.0,
                   'name': 'Potassium'},
    'Spironolactone': {'itemids': [50971], 'range': (1.5, 9.0), 'direction': 1, 'pos_t': 10.0, 'stable_t': 3.0,
                       'name': 'Potassium'},
    'Vancomycin': {'itemids': [50912], 'range': (0.1, 20.0), 'direction': 1, 'pos_t': 15.0, 'stable_t': 5.0,
                   'name': 'Creatinine'},
    'Gentamicin': {'itemids': [50912], 'range': (0.1, 20.0), 'direction': 1, 'pos_t': 20.0, 'stable_t': 5.0,
                   'name': 'Creatinine'},
    'Lisinopril': {'itemids': [50912], 'range': (0.1, 20.0), 'direction': 1, 'pos_t': 20.0, 'stable_t': 5.0,
                   'name': 'Creatinine'},
    'Mannitol': {'itemids': [50824], 'range': (200, 450), 'direction': 1, 'pos_t': 5.0, 'stable_t': 1.0,
                 'name': 'Osmolality'},
    'Hydrochlorothiazide': {'itemids': [50983], 'range': (100, 160), 'direction': -1, 'pos_t': 8.0, 'stable_t': 2.0,
                            'name': 'Sodium'},
    'Calcium Gluconate': {'itemids': [50893, 50808], 'range': (0.5, 15.0), 'direction': 1, 'pos_t': 8.0,
                          'stable_t': 2.0, 'name': 'Calcium'},
    'Magnesium Sulfate': {'itemids': [50960], 'range': (0.5, 10.0), 'direction': 1, 'pos_t': 15.0, 'stable_t': 5.0,
                          'name': 'Magnesium'},
    'Pantoprazole': {'itemids': [50960], 'range': (0.5, 10.0), 'direction': -1, 'pos_t': 10.0, 'stable_t': 3.0,
                     'name': 'Magnesium'},

    # 3. 血液与凝血
    'Heparin': {'itemids': [51275], 'range': (15, 150), 'direction': 1, 'pos_t': 25.0, 'stable_t': 8.0, 'name': 'PTT'},
    'Warfarin': {'itemids': [51237], 'range': (0.5, 15.0), 'direction': 1, 'pos_t': 20.0, 'stable_t': 5.0,
                 'name': 'INR'},
    'Phytonadione': {'itemids': [51237], 'range': (0.5, 15.0), 'direction': -1, 'pos_t': 20.0, 'stable_t': 5.0,
                     'name': 'INR'},
    'Enoxaparin': {'itemids': [51265], 'range': (5, 1000), 'direction': -1, 'pos_t': 30.0, 'stable_t': 10.0,
                   'name': 'Platelets'},
    'Albumin': {'itemids': [50862], 'range': (0.5, 6.0), 'direction': 1, 'pos_t': 15.0, 'stable_t': 5.0,
                'name': 'Albumin'},

    # 4. 感染与循环
    'Ceftriaxone': {'itemids': [51301], 'range': (0.1, 100), 'direction': -1, 'pos_t': 20.0, 'stable_t': 8.0,
                    'name': 'WBC'},
    'Dexamethasone': {'itemids': [51301], 'range': (0.1, 100), 'direction': 1, 'pos_t': 20.0, 'stable_t': 8.0,
                      'name': 'WBC'},
    'Norepinephrine': {'itemids': [50813], 'range': (0.1, 25.0), 'direction': -1, 'pos_t': 10.0, 'stable_t': 3.0,
                       'name': 'Lactate'},
    'Lactulose': {'itemids': [50866], 'range': (5, 500), 'direction': -1, 'pos_t': 20.0, 'stable_t': 8.0,
                  'name': 'Ammonia'},
    'Atorvastatin': {'itemids': [50861], 'range': (1, 1000), 'direction': 1, 'pos_t': 15.0, 'stable_t': 5.0,
                     'name': 'ALT'},
    'Amiodarone': {'itemids': [50863], 'range': (1, 1000), 'direction': 1, 'pos_t': 15.0, 'stable_t': 5.0,
                   'name': 'AST'},
    'Tacrolimus': {'itemids': [51006], 'range': (1, 150), 'direction': 1, 'pos_t': 15.0, 'stable_t': 5.0,
                   'name': 'BUN'},
    'Albuterol': {'itemids': [50971], 'range': (1.5, 9.0), 'direction': -1, 'pos_t': 8.0, 'stable_t': 2.0,
                  'name': 'Potassium'},
    'G-CSF': {'itemids': [51301], 'range': (0.1, 100), 'direction': 1, 'pos_t': 50.0, 'stable_t': 15.0, 'name': 'WBC'}
}

def build_refined_dataset():
    print("Step 1: 正在加载元数据与药物记录...")
    d_lab = pd.read_csv(os.path.join(BASE_PATH, "D_LABITEMS.csv"))
    itemid_to_label = d_lab.set_index('ITEMID')['LABEL'].to_dict()

    pres = pd.read_csv(os.path.join(BASE_PATH, "PRESCRIPTIONS.csv"), low_memory=False)
    pres['STARTDATE'] = pd.to_datetime(pres['STARTDATE'])
    pres['DOSE_VAL'] = pres['DOSE_VAL_RX'].fillna('')
    pres['DOSE_UNIT'] = pres['DOSE_UNIT_RX'].fillna('')

    # 筛选目标药物
    drug_patterns = list(EVIDENCE_MAP.keys())
    pattern = '|'.join(drug_patterns)
    target_drugs = pres[pres['DRUG'].str.contains(pattern, case=False, na=False)].copy()
    unique_hadms = target_drugs['HADM_ID'].unique()[:SAMPLE_LIMIT]
    target_drugs = target_drugs[target_drugs['HADM_ID'].isin(unique_hadms)]

    print(f"Step 2: 正在加载实验室指标 (涉及 {len(unique_hadms)} 个住院号)...")
    relevant_itemids = [iid for meta in EVIDENCE_MAP.values() for iid in meta['itemids']]
    labs_list = []
    # 使用 chunksize 读取大型实验室文件
    for chunk in pd.read_csv(os.path.join(BASE_PATH, "LABEVENTS.csv"), chunksize=1000000, low_memory=False):
        filtered = chunk[chunk['HADM_ID'].isin(unique_hadms) & chunk['ITEMID'].isin(relevant_itemids)]
        labs_list.append(filtered)
    labs = pd.concat(labs_list)
    labs['CHARTTIME'] = pd.to_datetime(labs['CHARTTIME'])
    labs = labs.dropna(subset=['VALUENUM'])

    dataset = []

    print("Step 3: 核心临床逻辑匹配中...")
    for hadm_id in tqdm(unique_hadms):
        p_drugs = target_drugs[target_drugs['HADM_ID'] == hadm_id]
        p_labs = labs[labs['HADM_ID'] == hadm_id]

        for _, drug_row in p_drugs.iterrows():
            d_name = drug_row['DRUG']
            d_time = drug_row['STARTDATE']
            d_dose = f"{drug_row['DOSE_VAL']} {drug_row['DOSE_UNIT']}"

            # 寻找匹配的配置
            current_config = None
            for key in EVIDENCE_MAP:
                if key.lower() in d_name.lower():
                    current_config = EVIDENCE_MAP[key]
                    break
            if not current_config: continue

            for iid in current_config['itemids']:
                spec_labs = p_labs[p_labs['ITEMID'] == iid]
                if spec_labs.empty: continue

                # 基线：给药前 24 小时内最后一次测量
                pre_recs = spec_labs[(spec_labs['CHARTTIME'] < d_time) &
                                     (spec_labs['CHARTTIME'] >= d_time - timedelta(hours=24))]
                if pre_recs.empty: continue
                v_pre = pre_recs.sort_values(by='CHARTTIME').iloc[-1]['VALUENUM']

                if not (current_config['range'][0] <= v_pre <= current_config['range'][1]): continue

                # 观测：给药后 24 小时内
                post_recs = spec_labs[(spec_labs['CHARTTIME'] > d_time) &
                                      (spec_labs['CHARTTIME'] <= d_time + timedelta(hours=24))]

                for _, post_row in post_recs.iterrows():
                    v_post = post_row['VALUENUM']
                    if not (current_config['range'][0] <= v_post <= current_config['range'][1]): continue

                    offset = (post_row['CHARTTIME'] - d_time).total_seconds() / 3600
                    delta_perc = ((v_post - v_pre) / v_pre) * 100

                    # --- 精细化标签逻辑 ---
                    exp_dir = current_config['direction']
                    pos_t = current_config['pos_t']
                    stable_t = current_config['stable_t']
                    abs_delta = abs(delta_perc)

                    if abs_delta >= pos_t:
                        # 变化大且方向一致 -> POS
                        if (exp_dir == 1 and delta_perc > 0) or (exp_dir == -1 and delta_perc < 0):
                            label = "POS"
                        else:
                            label = "OPPOSITE"  # 方向反转，具有临床研究价值
                    elif abs_delta <= stable_t:
                        label = "NEG"  # 波动极小
                    else:
                        label = "UNCERTAIN"

                    lab_label = itemid_to_label.get(iid, "Unknown Lab")

                    dataset.append({
                        'SUBJECT_ID': drug_row['SUBJECT_ID'],
                        'HADM_ID': hadm_id,
                        'DRUG': d_name,
                        'LAB_NAME': lab_label,
                        'BASE_VALUE': round(v_pre, 2),
                        'POST_VALUE': round(v_post, 2),
                        'DELTA_PERC': round(delta_perc, 2),
                        'TIME_OFFSET_HRS': round(offset, 2),
                        'LABEL': label,
                        'INPUT_TEXT': f"Patient was administered {d_name} ({d_dose}). Baseline {lab_label} was {v_pre}, and it changed to {v_post} ({delta_perc:+.1f}%) after {offset:.1f} hours."
                    })

    # Step 4: 去重与清理
    df_res = pd.DataFrame(dataset)
    if not df_res.empty:
        # 去重：同一住院、同一药物、同一指标在同一时间点的记录
        df_res = df_res.drop_duplicates(subset=['HADM_ID', 'DRUG', 'LAB_NAME', 'TIME_OFFSET_HRS'])
        print(f"\n处理完成！最终样本数: {len(df_res)}")
        df_res.to_csv(OUTPUT_FILE, index=False)
        print(f"数据已导出至: {OUTPUT_FILE}")
    else:
        print("未发现匹配样本。")


if __name__ == "__main__":
    build_refined_dataset()