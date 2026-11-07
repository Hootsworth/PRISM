"""Integration test for PRISM Demo Backend API."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient
from demo.backend.api import app

def main():
    client = TestClient(app)

    print("1. Testing /api/health...")
    r = client.get("/api/health")
    assert r.status_code == 200, f"Health check failed: {r.text}"
    health_data = r.json()
    print(f"   PASS: Status={health_data['status']}, Device={health_data['device']}, Checkpoint={health_data['checkpoint']}")

    print("2. Testing /api/samples...")
    r = client.get("/api/samples")
    assert r.status_code == 200, f"Samples failed: {r.text}"
    samples = r.json()["samples"]
    assert len(samples) > 0, "No samples returned"
    sample = samples[0]
    print(f"   PASS: Found {len(samples)} samples. Using sample: {sample['id']} ({sample['camera']}, ISO {sample['iso']})")

    print("3. Testing /api/sample_preview...")
    r = client.get(f"/api/sample_preview/{sample['id']}")
    assert r.status_code == 200
    print("   PASS: Sample preview generated")

    print("4. Testing B0 processing on sample...")
    b0_req = {
        "source_type": "sample",
        "source_id": sample["id"],
        "pipeline": "B0",
        "budget": 0.0,
        "tone_map": "reinhard",
        "max_preview_dim": 600
    }
    r = client.post("/api/process", json=b0_req)
    assert r.status_code == 200, f"B0 failed: {r.text}"
    b0_res = r.json()
    print(f"   PASS: B0 processed in {b0_res['elapsed_seconds']}s | PSNR: {b0_res['metrics']['psnr_db']} dB | SSIM: {b0_res['metrics']['ssim']}")

    print("5. Testing B2 (PRISM Hybrid, budget=0.05) on sample...")
    b2_req = {
        "source_type": "sample",
        "source_id": sample["id"],
        "pipeline": "B2",
        "budget": 0.05,
        "tone_map": "reinhard",
        "max_preview_dim": 600
    }
    r = client.post("/api/process", json=b2_req)
    assert r.status_code == 200, f"B2 failed: {r.text}"
    b2_res = r.json()
    print(f"   PASS: B2 processed in {b2_res['elapsed_seconds']}s | PSNR: {b2_res['metrics']['psnr_db']} dB | SSIM: {b2_res['metrics']['ssim']} | Tiles refined: {b2_res['stats']['tiles_refined']} ({b2_res['stats']['refinement_percentage']}%)")
    assert b2_res["heatmap_image"] is not None, "Heatmap missing"
    assert b2_res["overlay_image"] is not None, "Overlay missing"

    print("6. Testing B1 (Neural) on sample...")
    b1_req = {
        "source_type": "sample",
        "source_id": sample["id"],
        "pipeline": "B1",
        "budget": 1.0,
        "tone_map": "reinhard",
        "max_preview_dim": 600
    }
    r = client.post("/api/process", json=b1_req)
    assert r.status_code == 200, f"B1 failed: {r.text}"
    b1_res = r.json()
    print(f"   PASS: B1 processed in {b1_res['elapsed_seconds']}s | PSNR: {b1_res['metrics']['psnr_db']} dB | SSIM: {b1_res['metrics']['ssim']}")

    print("7. Testing /api/download...")
    r = client.get(b2_res["download_url"])
    assert r.status_code == 200
    assert len(r.content) > 1000
    print(f"   PASS: Image downloaded ({len(r.content)} bytes)")

    print("8. Testing Static Frontend index.html...")
    r = client.get("/")
    assert r.status_code == 200
    assert "PRISM" in r.text
    print("   PASS: Frontend HTML served properly at root /")

    print("\nALL 8/8 INTEGRATION TESTS PASSED SUCCESSFULLY!")

if __name__ == '__main__':
    main()
