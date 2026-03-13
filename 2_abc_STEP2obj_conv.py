import os, shutil, multiprocessing
import open3d as o3d
from pathlib import Path
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
import numpy as np

def step_to_obj_conversion(step_path, save_path, return_dict):
    """STEP 파일을 읽어 법선 방향이 교정된 OBJ 파일로 저장하는 함수 (멀티프로세싱용)"""
    try:
        reader = STEPControl_Reader()
        if reader.ReadFile(str(step_path)) != 1:
            return_dict['success'] = False
            return
        
        reader.TransferRoots()
        shape = reader.OneShape()
        
        # 1. 메시 생성 (정밀도 0.1)
        BRepMesh_IncrementalMesh(shape, 0.1).Perform()
        
        verts, tris = [], []
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        
        while explorer.More():
            face = explorer.Current()
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation(face, loc)
            
            if tri:
                v_offset = len(verts)
                trans = loc.Transformation()
                
                # 정점 추출 (변환 행렬 적용)
                for i in range(1, tri.NbNodes() + 1):
                    p = tri.Node(i).Transformed(trans)
                    verts.append([p.X(), p.Y(), p.Z()])
                
                # 삼각형 인덱스 추출 시 면의 방향성(Orientation) 반영
                is_reversed = (face.Orientation() == TopAbs_REVERSED)
                for i in range(1, tri.NbTriangles() + 1):
                    t = tri.Triangle(i)
                    n1, n2, n3 = t.Get()
                    
                    if is_reversed:
                        # 면이 반전된 경우 인덱스 순서를 바꿔서 감김 방향을 교정
                        tris.append([n1-1+v_offset, n3-1+v_offset, n2-1+v_offset])
                    else:
                        tris.append([n1-1+v_offset, n2-1+v_offset, n3-1+v_offset])
            explorer.Next()
        
        if not verts:
            return_dict['success'] = False
            return

        # 2. Open3D Mesh 생성 및 기하 구조 교정
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.array(verts))
        mesh.triangles = o3d.utility.Vector3iVector(np.array(tris))
        
        # 중복 데이터 제거
        mesh = mesh.remove_duplicated_vertices()
        mesh = mesh.remove_unreferenced_vertices()
        
        # 모든 삼각형의 방향을 외부를 향하도록 일관되게 정렬
        mesh.orient_triangles()
        
        # 법선 재계산
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        
        # 3. OBJ 저장
        o3d.io.write_triangle_mesh(str(save_path), mesh)
        
        return_dict['success'] = True

    except Exception as e:
        print(f"변환 중 에러 발생 ({step_path.name}): {e}")
        return_dict['success'] = False

def process_chunk(base_dir, chunk_idx, timeout_sec=180):
    """지정된 청크 폴더(0000~0099) 내의 모든 모델을 변환하고 타임아웃 시 폴더 삭제"""
    chunk_name = str(chunk_idx).zfill(4)
    chunk_path = Path(base_dir) / chunk_name
    
    if not chunk_path.exists():
        print(f"청크 폴더를 찾을 수 없습니다: {chunk_path}")
        return

    model_folders = [f for f in chunk_path.iterdir() if f.is_dir()]
    print(f"--- 청크 {chunk_name} 변환 시작 (총 {len(model_folders)}개 모델) ---")
    
    success_count = 0
    deleted_count = 0
    
    for model_folder in model_folders:
        step_files = list(model_folder.glob("*.step"))
        if not step_files:
            continue
            
        step_file = step_files[0]
        obj_save_path = model_folder / (step_file.stem + ".obj")
        
        # 이미 OBJ가 존재하면 건너뛰기
        if obj_save_path.exists():
            success_count += 1
            continue
            
        # 멀티프로세싱 상태 공유를 위한 딕셔너리
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        return_dict['success'] = False
        
        # 변환 작업을 별도의 프로세스로 실행
        p = multiprocessing.Process(target=step_to_obj_conversion, args=(step_file, obj_save_path, return_dict))
        p.start()
        
        # 타임아웃(3분=180초)만큼 대기
        p.join(timeout=timeout_sec)
        
        if p.is_alive():
            # 3분이 지났는데도 프로세스가 살아있다면 무한 루프에 빠진 것으로 간주
            print(f" -> 타임아웃 ({timeout_sec}초 초과): {step_file.stem} 프로세스 강제 종료 및 폴더 삭제 중...")
            p.terminate()
            p.join()
            
            # 해당 모델 폴더 전체를 디스크에서 삭제
            try:
                shutil.rmtree(model_folder)
                deleted_count += 1
            except Exception as e:
                print(f"폴더 삭제 실패 ({model_folder}): {e}")
            continue
            
        # 정상적으로 프로세스가 끝난 경우 결과 확인
        if return_dict.get('success', False):
            success_count += 1
            if success_count % 10 == 0:
                print(f"청크 {chunk_name} 진행 중: {success_count}/{len(model_folders)}")
        else:
            print(f" -> 변환 실패 (에러 또는 빈 형상): {step_file.stem}")

    print(f"--- 청크 {chunk_name} 완료 (성공: {success_count}, 시간초과 삭제: {deleted_count}) ---")

# ==========================================
# 실행부 (멀티프로세싱 안전 구문)
# ==========================================
if __name__ == "__main__":
    BASE_DIRECTORY = "./abc_dataset_filtered-1"
    TIMEOUT_LIMIT = 180 # 3분 (180초)

    # 예: 1번 청크부터 20번 청크까지 실행
    for i in range(17, 25): 
        process_chunk(BASE_DIRECTORY, i, timeout_sec=TIMEOUT_LIMIT)