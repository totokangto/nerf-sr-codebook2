import os
import json
import pandas as pd

# 데이터를 저장할 빈 리스트 생성
data_list = []

# 파일이 있는 디렉토리 경로 설정
directory = '/home/sjchoi/Downloads/psnr_refine_val'  # 실제 경로로 변경하세요


# 디렉토리 내의 모든 파일을 순회
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        # 파일명에서 PSNR 컬럼명 생성
        psnr_column = filename.replace('.json', '').split('ds2')[-1][1:]
        
        # 파일 경로 생성
        file_path = os.path.join(directory, filename)
        
        # JSON 파일 열기 및 데이터 읽기
        with open(file_path, 'r') as file:
            data = json.load(file)

        df = pd.DataFrame(data, columns=['Timestamp', 'Epoch', 'PSNR'])

        # 'Epoch'이 40000부터 46000 사이에 있는 데이터만 필터링
        filtered_df = df[(df['Epoch'] >= 40000) & (df['Epoch'] <= 46000)]
        
        # 각 필터링된 데이터에 대해
        for _, row in filtered_df.iterrows():
            # 데이터 복사 및 새로운 컬럼 추가
            entry = row.to_dict()  # 행 데이터를 딕셔너리로 변환
            print(entry)
            entry[psnr_column] = entry.pop('PSNR')  # PSNR 값을 새로운 컬럼명으로 이동
            data_list.append(entry)  # 리스트에 추가

        for item in data_list:
            print(item)

# 데이터프레임 생성
df = pd.DataFrame(data_list)

# 중복 컬럼이 있다면 하나로 통합
df = df.groupby(['Epoch'], as_index=False).first()

# 결과 출력 또는 파일로 저장
print(df)

# 필요에 따라 파일로 저장
df.to_csv('merged_data.csv', index=False)
