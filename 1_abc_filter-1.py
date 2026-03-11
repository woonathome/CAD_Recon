import os
import shutil
import multiprocessing
import py7zr
from pathlib import Path

# OpenCASCADE 라이브러리 임포트
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID
from OCC.Core.BRepCheck import BRepCheck_Analyzer

def check_single_watertight_body(step_path, return_dict):
    """
    STEP 파일을 읽어 1차(단일 Body) 및 2차(Watertight) 필터링을 수행.
    """
    try:
        reader = STEPControl_Reader()
        status = reader.ReadFile(str(step_path))
        
        if status != 1:
            return_dict['result'] = False
            return
            
        reader.TransferRoots()
        shape = reader.OneShape()
        
        if shape.IsNull():
            return_dict['result'] = False
            return

        # 1차: 단일 Solid 검사
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        solid_count = 0
        while explorer.More():
            solid_count += 1
            explorer.Next()
            
        if solid_count != 1:
            return_dict['result'] = False
            return

        # 2차: Watertight 무결성 검사
        analyzer = BRepCheck_Analyzer(shape)
        if not analyzer.IsValid():
            return_dict['result'] = False
            return

        return_dict['result'] = True
        
    except Exception as e:
        return_dict['result'] = False

def process_single_chunk(step_7z_path, meta_7z_path, output_dir_path, temp_dir_path, start_idx=0, timeout_sec=180):
    """
    지정된 단일 청크 쌍만 압축 해제(py7zr 사용) 및 필터링하는 파이프라인.
    """
    step_archive = Path(step_7z_path)
    meta_archive = Path(meta_7z_path)
    temp_path = Path(temp_dir_path)
    output_path = Path(output_dir_path)
    
    # 결과 폴더 생성
    output_path.mkdir(parents=True, exist_ok=True)

    # 청크 고유 이름 추출 (예: "abc_0000_step_v00.7z" -> "abc_0000")
    chunk_basename = step_archive.name.split('_step_')[0]
    marker_file = temp_path / ".current_chunk"

    # 1. 임시 폴더 상태 확인 및 스마트 Unzip
    need_unzip = True
    if temp_path.exists() and marker_file.exists():
        with open(marker_file, 'r') as f:
            if f.read().strip() == chunk_basename:
                need_unzip = False  # 이름표가 일치하면 Unzip 건너뜀

    if need_unzip:
        print(f"폴더 비우기 및 [{chunk_basename}] 청크 압축 해제 시작 (py7zr 사용)...")
        if temp_path.exists():
            shutil.rmtree(temp_path)
        temp_path.mkdir(parents=True, exist_ok=True)

        # py7zr을 이용한 압축 해제
        print(f"[{step_archive.name}] 압축 해제 중...")
        with py7zr.SevenZipFile(step_archive, mode='r') as z:
            z.extractall(path=temp_path)
            
        print(f"[{meta_archive.name}] 압축 해제 중...")
        with py7zr.SevenZipFile(meta_archive, mode='r') as z:
            z.extractall(path=temp_path)

        # 압축 해제 완료 후 이름표 쓰기
        with open(marker_file, 'w') as f:
            f.write(chunk_basename)
    else:
        print(f"[{chunk_basename}] 청크가 이미 준비되어 있습니다. Unzip을 건너뜁니다.")

    # 2. 필터링 및 복사 로직 (start_idx부터)
    print(f"\n필터링 시작 (시작 번호: {start_idx}번부터)")
    for model_dir in sorted(temp_path.iterdir()):
        if not model_dir.is_dir():
            continue
            
        model_id = model_dir.name
        
        # 지정된 시작 번호 이전 모델은 스킵
        try:
            model_num = int(model_id)
            if model_num < start_idx:
                continue
        except ValueError:
            continue
            
        step_files = list(model_dir.glob("*.step"))
        yml_files = list(model_dir.glob("*.yml"))
        
        if not step_files or not yml_files:
            continue
            
        step_file = step_files[0]
        yml_file = yml_files[0]
        
        print(f"모델 [{model_id}] 검사 중...")
        
        # 멀티프로세싱을 이용한 타임아웃 적용
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        return_dict['result'] = False 
        
        p = multiprocessing.Process(target=check_single_watertight_body, args=(step_file, return_dict))
        p.start()
        p.join(timeout=timeout_sec)
        
        if p.is_alive():
            print(f" -> X (시간 초과: {timeout_sec}초 강제 종료)")
            p.terminate()
            p.join()
            continue
            
        if return_dict['result']:
            print(f" -> O 저장 중...")
            
            # --- 변경된 부분: 계층형 폴더 분산 로직 ---
            # 모델 번호의 앞 4자리를 상위 폴더 이름으로 지정 (예: 00012345 -> 0001)
            prefix_folder = model_id[:4] 
            target_model_dir = output_path / prefix_folder / model_id
            
            target_model_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(step_file, target_model_dir / step_file.name)
            shutil.copy2(yml_file, target_model_dir / yml_file.name)
            # -----------------------------------------
            
        else:
            print(f" -> X (조건 불만족)")

    print(f"\n[{chunk_basename}] 단일 청크 처리가 완료되었습니다!")

# ==========================================
# 실행 부분
# ==========================================
if __name__ == "__main__":
    # 이번에 실행할 타겟 파일 지정
    CHUNK_NUM = 7
    CHUNK_str = str.zfill(str(CHUNK_NUM), 4)

    STEP_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_step_v00.7z"
    META_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_meta_v00.7z"
    
    OUTPUT_DIRECTORY = "./abc_dataset_filtered-1"
    TEMP_DIRECTORY = "./temp_abc_extract"
    
    # 설정값
    # START_INDEX = 10000 * CHUNK_NUM     # 시작 번호
    START_INDEX = 78991     # 시작 번호 지정
    TIMEOUT_SECONDS = 180  # 모델당 3분 대기
    
    process_single_chunk(STEP_7Z_FILE, META_7Z_FILE, OUTPUT_DIRECTORY, TEMP_DIRECTORY, START_INDEX, TIMEOUT_SECONDS)



    # 이번에 실행할 타겟 파일 지정
    CHUNK_NUM = 8
    CHUNK_str = str.zfill(str(CHUNK_NUM), 4)

    STEP_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_step_v00.7z"
    META_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_meta_v00.7z"
    
    OUTPUT_DIRECTORY = "./abc_dataset_filtered-1"
    TEMP_DIRECTORY = "./temp_abc_extract"
    
    # 설정값
    START_INDEX = 10000 * CHUNK_NUM     # 시작 번호
    TIMEOUT_SECONDS = 180  # 모델당 3분 대기
    
    process_single_chunk(STEP_7Z_FILE, META_7Z_FILE, OUTPUT_DIRECTORY, TEMP_DIRECTORY, START_INDEX, TIMEOUT_SECONDS)


    
    # 이번에 실행할 타겟 파일 지정
    CHUNK_NUM = 9
    CHUNK_str = str.zfill(str(CHUNK_NUM), 4)

    STEP_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_step_v00.7z"
    META_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_meta_v00.7z"
    
    OUTPUT_DIRECTORY = "./abc_dataset_filtered-1"
    TEMP_DIRECTORY = "./temp_abc_extract"
    
    # 설정값
    START_INDEX = 10000 * CHUNK_NUM     # 시작 번호
    TIMEOUT_SECONDS = 180  # 모델당 3분 대기
    
    process_single_chunk(STEP_7Z_FILE, META_7Z_FILE, OUTPUT_DIRECTORY, TEMP_DIRECTORY, START_INDEX, TIMEOUT_SECONDS)
    

    
    # 이번에 실행할 타겟 파일 지정
    CHUNK_NUM = 10
    CHUNK_str = str.zfill(str(CHUNK_NUM), 4)

    STEP_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_step_v00.7z"
    META_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_meta_v00.7z"
    
    OUTPUT_DIRECTORY = "./abc_dataset_filtered-1"
    TEMP_DIRECTORY = "./temp_abc_extract"
    
    # 설정값
    START_INDEX = 10000 * CHUNK_NUM     # 시작 번호
    TIMEOUT_SECONDS = 180  # 모델당 3분 대기
    
    process_single_chunk(STEP_7Z_FILE, META_7Z_FILE, OUTPUT_DIRECTORY, TEMP_DIRECTORY, START_INDEX, TIMEOUT_SECONDS)
    

    
    # 이번에 실행할 타겟 파일 지정
    CHUNK_NUM = 11
    CHUNK_str = str.zfill(str(CHUNK_NUM), 4)

    STEP_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_step_v00.7z"
    META_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_meta_v00.7z"
    
    OUTPUT_DIRECTORY = "./abc_dataset_filtered-1"
    TEMP_DIRECTORY = "./temp_abc_extract"
    
    # 설정값
    START_INDEX = 10000 * CHUNK_NUM     # 시작 번호
    TIMEOUT_SECONDS = 180  # 모델당 3분 대기
    
    process_single_chunk(STEP_7Z_FILE, META_7Z_FILE, OUTPUT_DIRECTORY, TEMP_DIRECTORY, START_INDEX, TIMEOUT_SECONDS)
    

    
    # 이번에 실행할 타겟 파일 지정
    CHUNK_NUM = 12
    CHUNK_str = str.zfill(str(CHUNK_NUM), 4)

    STEP_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_step_v00.7z"
    META_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_meta_v00.7z"
    
    OUTPUT_DIRECTORY = "./abc_dataset_filtered-1"
    TEMP_DIRECTORY = "./temp_abc_extract"
    
    # 설정값
    START_INDEX = 10000 * CHUNK_NUM     # 시작 번호
    TIMEOUT_SECONDS = 180  # 모델당 3분 대기
    
    process_single_chunk(STEP_7Z_FILE, META_7Z_FILE, OUTPUT_DIRECTORY, TEMP_DIRECTORY, START_INDEX, TIMEOUT_SECONDS)
    

    
    # 이번에 실행할 타겟 파일 지정
    CHUNK_NUM = 13
    CHUNK_str = str.zfill(str(CHUNK_NUM), 4)

    STEP_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_step_v00.7z"
    META_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_meta_v00.7z"
    
    OUTPUT_DIRECTORY = "./abc_dataset_filtered-1"
    TEMP_DIRECTORY = "./temp_abc_extract"
    
    # 설정값
    START_INDEX = 10000 * CHUNK_NUM     # 시작 번호
    TIMEOUT_SECONDS = 180  # 모델당 3분 대기
    
    process_single_chunk(STEP_7Z_FILE, META_7Z_FILE, OUTPUT_DIRECTORY, TEMP_DIRECTORY, START_INDEX, TIMEOUT_SECONDS)
    

    
    # 이번에 실행할 타겟 파일 지정
    CHUNK_NUM = 14
    CHUNK_str = str.zfill(str(CHUNK_NUM), 4)

    STEP_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_step_v00.7z"
    META_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_meta_v00.7z"
    
    OUTPUT_DIRECTORY = "./abc_dataset_filtered-1"
    TEMP_DIRECTORY = "./temp_abc_extract"
    
    # 설정값
    START_INDEX = 10000 * CHUNK_NUM     # 시작 번호
    TIMEOUT_SECONDS = 180  # 모델당 3분 대기
    
    process_single_chunk(STEP_7Z_FILE, META_7Z_FILE, OUTPUT_DIRECTORY, TEMP_DIRECTORY, START_INDEX, TIMEOUT_SECONDS)
    

    
    # 이번에 실행할 타겟 파일 지정
    CHUNK_NUM = 15
    CHUNK_str = str.zfill(str(CHUNK_NUM), 4)

    STEP_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_step_v00.7z"
    META_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_meta_v00.7z"
    
    OUTPUT_DIRECTORY = "./abc_dataset_filtered-1"
    TEMP_DIRECTORY = "./temp_abc_extract"
    
    # 설정값
    START_INDEX = 10000 * CHUNK_NUM     # 시작 번호
    TIMEOUT_SECONDS = 180  # 모델당 3분 대기
    
    process_single_chunk(STEP_7Z_FILE, META_7Z_FILE, OUTPUT_DIRECTORY, TEMP_DIRECTORY, START_INDEX, TIMEOUT_SECONDS)

        # 이번에 실행할 타겟 파일 지정
    CHUNK_NUM = 16
    CHUNK_str = str.zfill(str(CHUNK_NUM), 4)

    STEP_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_step_v00.7z"
    META_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_meta_v00.7z"
    
    OUTPUT_DIRECTORY = "./abc_dataset_filtered-1"
    TEMP_DIRECTORY = "./temp_abc_extract"
    
    # 설정값
    START_INDEX = 10000 * CHUNK_NUM     # 시작 번호
    TIMEOUT_SECONDS = 180  # 모델당 3분 대기
    
    process_single_chunk(STEP_7Z_FILE, META_7Z_FILE, OUTPUT_DIRECTORY, TEMP_DIRECTORY, START_INDEX, TIMEOUT_SECONDS)


    # 이번에 실행할 타겟 파일 지정
    CHUNK_NUM = 17
    CHUNK_str = str.zfill(str(CHUNK_NUM), 4)

    STEP_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_step_v00.7z"
    META_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_meta_v00.7z"
    
    OUTPUT_DIRECTORY = "./abc_dataset_filtered-1"
    TEMP_DIRECTORY = "./temp_abc_extract"
    
    # 설정값
    START_INDEX = 10000 * CHUNK_NUM     # 시작 번호
    TIMEOUT_SECONDS = 180  # 모델당 3분 대기
    
    process_single_chunk(STEP_7Z_FILE, META_7Z_FILE, OUTPUT_DIRECTORY, TEMP_DIRECTORY, START_INDEX, TIMEOUT_SECONDS)


        # 이번에 실행할 타겟 파일 지정
    CHUNK_NUM = 18
    CHUNK_str = str.zfill(str(CHUNK_NUM), 4)

    STEP_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_step_v00.7z"
    META_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_meta_v00.7z"
    
    OUTPUT_DIRECTORY = "./abc_dataset_filtered-1"
    TEMP_DIRECTORY = "./temp_abc_extract"
    
    # 설정값
    START_INDEX = 10000 * CHUNK_NUM     # 시작 번호
    TIMEOUT_SECONDS = 180  # 모델당 3분 대기
    
    process_single_chunk(STEP_7Z_FILE, META_7Z_FILE, OUTPUT_DIRECTORY, TEMP_DIRECTORY, START_INDEX, TIMEOUT_SECONDS)

        # 이번에 실행할 타겟 파일 지정
    CHUNK_NUM = 19
    CHUNK_str = str.zfill(str(CHUNK_NUM), 4)

    STEP_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_step_v00.7z"
    META_7Z_FILE = f"./abc_dataset_download/abc_{CHUNK_str}_meta_v00.7z"
    
    OUTPUT_DIRECTORY = "./abc_dataset_filtered-1"
    TEMP_DIRECTORY = "./temp_abc_extract"
    
    # 설정값
    START_INDEX = 10000 * CHUNK_NUM     # 시작 번호
    TIMEOUT_SECONDS = 180  # 모델당 3분 대기
    
    process_single_chunk(STEP_7Z_FILE, META_7Z_FILE, OUTPUT_DIRECTORY, TEMP_DIRECTORY, START_INDEX, TIMEOUT_SECONDS)