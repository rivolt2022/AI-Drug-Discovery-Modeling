"""
모든 테스트 실행 스크립트

이 스크립트는 모든 테스트를 순차적으로 실행합니다.
"""

import sys
import os
import subprocess
import time

def run_test_script(script_name, description):
    """개별 테스트 스크립트 실행"""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        print(f"실행 시간: {end_time - start_time:.2f}초")
        
        if result.returncode == 0:
            print("✅ 테스트 성공")
            print("출력:")
            print(result.stdout)
            return True
        else:
            print("❌ 테스트 실패")
            print("오류:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 테스트 시간 초과 (5분)")
        return False
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류: {e}")
        return False

def main():
    """메인 함수"""
    print("🚀 AI 신약개발 데이터 로딩 전체 테스트 시작")
    print("=" * 60)
    
    # 테스트 스크립트 목록
    tests = [
        ("test_quick_check.py", "빠른 데이터 확인"),
        ("test_data_loading.py", "데이터 로딩 단위 테스트"),
        ("test_data_integration.py", "데이터 통합 테스트")
    ]
    
    results = []
    
    for script_name, description in tests:
        success = run_test_script(script_name, description)
        results.append((description, success))
    
    # 결과 요약
    print(f"\n{'='*60}")
    print("📊 테스트 결과 요약")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for description, success in results:
        status = "✅ 통과" if success else "❌ 실패"
        print(f"{description}: {status}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\n총 테스트: {len(results)}개")
    print(f"통과: {passed}개")
    print(f"실패: {failed}개")
    
    if failed == 0:
        print("\n�� 모든 테스트가 성공적으로 완료되었습니다!")
        return True
    else:
        print(f"\n⚠️ {failed}개의 테스트가 실패했습니다.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
