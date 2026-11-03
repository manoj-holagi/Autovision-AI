from backend.risk import CollisionRiskEngine


def test_risk_is_unavailable_without_vehicle_evidence():
    result = CollisionRiskEngine().update(0.0, 640, 480, [], 25)
    assert result["status"] == "UNAVAILABLE"
    assert "not evidence that the road is clear" in result["reason"]
    assert result["estimated_time_to_collision_seconds"] is None


def test_risk_requires_temporal_large_centered_growth_and_never_claims_collision():
    engine = CollisionRiskEngine()
    results = []
    boxes = [[280, 180, 360, 260], [260, 160, 380, 280], [230, 125, 410, 305]]
    for i, box in enumerate(boxes):
        results.append(engine.update(i * .3, 640, 480, [{"label": "car", "confidence": .9, "bbox": box}], 40))
    assert results[0]["alert"] is None
    assert results[-1]["alert"]["type"] == "POTENTIAL_FORWARD_OBSTACLE"
    assert results[-1]["estimated_time_to_collision_seconds"] is None
    assert "High." in results[-1]["uncertainty"]


def test_risk_rejects_stale_temporal_observations():
    engine = CollisionRiskEngine()
    det = [{"label": "car", "confidence": .9, "bbox": [250, 150, 390, 300]}]
    engine.update(0, 640, 480, det, 20)
    stale = engine.update(2, 640, 480, det, 20)
    assert stale["status"] == "UNAVAILABLE"
    assert "gap" in stale["reason"]


def test_risk_does_not_track_low_confidence_detections():
    engine=CollisionRiskEngine(minimum_confidence=.35)
    detection={"label":"car","confidence":.2,"bbox":[250,150,390,300]}
    result=engine.update(0.0,640,480,[detection],20)
    assert result["status"]=="UNAVAILABLE"
    assert engine.tracks==[]

