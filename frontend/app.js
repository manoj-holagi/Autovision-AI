const $ = selector => document.querySelector(selector);
const MAX_UPLOAD_BYTES = 250 * 1024 * 1024;
const MAX_INFERENCE_FRAME_EDGE = 1280;
const state = { page: 1, pageSize: 8, searchTimer: null };

async function api(path, options) {
  const response = await fetch(path, options);
  let body;
  try { body = await response.json(); } catch { body = { detail: response.statusText }; }
  if (!response.ok) throw Error(body.detail || `Request failed (${response.status})`);
  return body;
}
function esc(value) {
  return String(value).replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[char]));
}
function mediaUrl(url) { return url.startsWith('http') ? url : `${location.origin}${url}`; }
function countLabel(count) { return `${count} detection instance${count === 1 ? '' : 's'}`; }
function classBreakdown(classes = {}) {
  const parts = Object.entries(classes).map(([name, count]) => `${esc(name)} ${count}`);
  return parts.length ? parts.join(' · ') : 'No detections';
}
function completionLabel(status, item = {}) {
  if (status === 'stopped') return 'Stopped — partial simulated replay';
  if (status === 'error') return 'Stopped after a playback or inference error';
  if (status === 'complete' && Number.isFinite(item.sampling_rate_hz)) return 'Complete — browser playback reached end';
  if (status === 'complete') return 'Complete — decoded source frame count matched';
  if (status === 'partial') return 'Partial — decoder stopped before the source frame count';
  return 'Finished — full source completion unverified';
}
function duration(value) { return Number.isFinite(value) ? `${value.toFixed(2)} s` : 'Not recorded'; }
function videoSummary(item) {
  const sampled = Number.isInteger(item.sampled_frame_count)
    ? `${item.sampled_frame_count}`
    : `Not recorded${item.detection_timestamps ? ` (detections on ${item.detection_timestamps} distinct timestamps)` : ''}`;
  const metric = (label, value) => `<div class="summary-metric"><dt>${label}</dt><dd>${value}</dd></div>`;
  const frameLabel = Number.isFinite(item.sampling_rate_hz) ? 'Browser frames presented' : 'Frames processed';
  const modules = item.module_results || [];
  const driver = modules.map(x=>x.drowsiness).filter(Boolean);
  const lanes = modules.map(x=>x.lane).filter(Boolean);
  const risks = modules.map(x=>x.collision_risk).filter(Boolean);
  const potholes=modules.map(x=>x.pothole).filter(Boolean);
  const profileAssessment=modules.find(x=>x.profile)?.profile;
  const potholeEvents=modules.filter(x=>x.pothole_event).length;
  const moduleSummary = [potholes.length ? `Pothole model: ${potholes.reduce((sum,x)=>sum+(x.detection_count||0),0)} candidate instances, ${potholeEvents} deduplicated event(s)` : '', driver.length ? `Driver monitor: ${driver.filter(x=>x.indicator==='POSSIBLE_PROLONGED_EYE_CLOSURE').length} prolonged-closure indicator(s), ${driver.filter(x=>x.status==='FACE_NOT_VISIBLE').length} no-face sample(s)` : '', lanes.length ? `Lane baseline: ${lanes.filter(x=>x.status==='ESTIMATED').length}/${lanes.length} sampled estimate(s)` : '', risks.length ? `Risk indicator: ${risks.filter(x=>x.alert).length} experimental warning(s)` : ''].filter(Boolean).join(' · ') || 'No additional video modules recorded';
  return `<details class="video-summary" open><summary>Video processing summary <span class="status-${esc(item.completion_status || 'finished_unverified')}">${esc(completionLabel(item.completion_status,item))}</span></summary><dl>${metric(frameLabel, item.frames_processed ?? item.frame_count)}${metric('Frames sampled', sampled)}${metric('Total detection instances', item.total_detections ?? item.detections?.length ?? 0)}${metric('Detections by class', classBreakdown(item.detections_by_class))}${metric('Average processing time / sampled frame', Number.isFinite(item.average_inference_ms) ? `${item.average_inference_ms.toFixed(1)} ms` : 'Not recorded for this run')}${metric('Total processing time', duration(item.processing_time_seconds))}${metric('Source frames reported by video', item.source_frame_count ?? 'Not recorded for this run')}${metric('Model used', item.model_used || 'Not recorded for this run')}${metric('Selected video profile', profileAssessment?.profile?.replaceAll('_',' ') || 'Not recorded for this run')}${metric('Observed video cues', profileAssessment ? `Road: ${esc(profileAssessment.road_scene)} (${profileAssessment.road_evidence_frames} sample(s)); driver face: ${esc(profileAssessment.driver_face)} (${profileAssessment.driver_face_visible_samples} sample(s))` : 'Not recorded for this run')}${metric('Confidence threshold', Number.isFinite(item.confidence_threshold) ? item.confidence_threshold.toFixed(2) : 'Not recorded for this run')}${metric('Sampling interval', Number.isInteger(item.frame_interval) ? `Every ${item.frame_interval} frames` : Number.isFinite(item.sampling_rate_hz) ? `${item.sampling_rate_hz} samples/s` : 'Not recorded for this run')}${metric('Video modules', moduleSummary)}</dl></details>`;
}
async function refreshStats() {
  const health = await api('/api/health');
  $('.connection span').textContent = `API ${health.status.toUpperCase()} · ${health.database}`;
}
async function loadModels() {
  const data = await api('/api/detectors');
  $('#model-list').innerHTML = data.items.map(model => `<div class="model"><span class="model-name">${esc(model.name)}</span><span class="badge ${esc(model.status)}">${esc(model.status.replaceAll('_', ' '))}</span><small>${esc(model.limitation)}${model.weights ? ` · ${esc(model.weights)}` : ''} · Runtime: ${esc((model.runtime_status||'not checked').replaceAll('_',' '))}</small></div>`).join('');
  const capabilityIds = {vehicle:'vehicle',pothole:'pothole',drowsiness:'driver',lane:'lane',crash_risk:'risk'};
  for (const model of data.items) {
    const target = capabilityIds[model.id];
    if (!target) continue;
    const runtime = (model.runtime_status || 'not checked').replaceAll('_',' ');
    setCapability(target, model.status.toUpperCase(), `${runtime} · ${model.limitation}`);
  }
}
async function loadEvents() {
  try {
    const params = new URLSearchParams({ page: state.page, page_size: state.pageSize, search: $('#search').value, detector: $('#filter').value });
    const data = await api(`/api/events?${params}`);
    $('#events-list').innerHTML = data.items.length ? data.items.map(item => {
      const count = item.total_detections ?? item.detections.length;
      const countText = item.media_type === 'video' ? countLabel(count) : countLabel(count);
      const frameNoun = Number.isFinite(item.sampling_rate_hz) ? 'frames presented' : 'frames processed';
      return `<article class="event-card"><div class="event"><img class="thumb" src="${esc(mediaUrl(item.annotated_url))}" alt="Annotated preview"><div><div class="event-title">${esc(item.filename)}</div><div class="event-meta">${new Date(item.created_at).toLocaleString()} · ${esc(item.detector)} · ${item.frames_processed ?? item.frame_count} ${frameNoun}</div></div><span class="event-count">${esc(countText)}</span></div>${item.media_type === 'video' ? videoSummary(item) : ''}</article>`;
    }).join('') : '<div class="empty">No analyses yet. Upload an image or video to begin.</div>';
    $('#page-info').textContent = `${data.total ? (state.page - 1) * state.pageSize + 1 : 0}–${Math.min(state.page * state.pageSize, data.total)} of ${data.total}`;
    $('#prev').disabled = state.page <= 1;
    $('#next').disabled = state.page * state.pageSize >= data.total;
  } catch (error) {
    $('#events-list').innerHTML = `<div class="empty">${esc(error.message)}</div>`;
  }
}
$('#refresh').onclick = refreshAll;
$('#reload-models').onclick = loadModels;
$('#filter').onchange = () => { state.page = 1; loadEvents(); };
$('#search').oninput = () => { clearTimeout(state.searchTimer); state.searchTimer = setTimeout(() => { state.page = 1; loadEvents(); }, 250); };
$('#prev').onclick = () => { state.page = Math.max(1, state.page - 1); loadEvents(); };
$('#next').onclick = () => { state.page++; loadEvents(); };

// Simulated live replay: browser playback is local; only individually sampled frames
// are sent to the backend, sequentially, with their exact media timestamp.
const sim = { file: null, url: null, sessionId: null, active: false, starting: false, paused: true, busy: false,
  lastSampleTime: -1, frameId: 0, presented: 0, detections: [], classes: {}, alerts: [], callback: null,
  startedAt: 0, inferenceMs: 0, samples: 0, skippedSamples: 0, vehicleInstances: 0, potholeEvents: 0, driverAlerts: 0,
  laneAvailableSamples: 0, riskAlerts: 0, lastRiskTimestamp: -Infinity, completed: false, pendingFinish: null };
const video = $('#sim-video'), overlay = $('#sim-overlay'), ctx = overlay.getContext('2d');
const fmtTime = seconds => `${Math.floor((seconds || 0) / 60)}:${String(Math.floor((seconds || 0) % 60)).padStart(2, '0')}`;
function setSimState(label, cls = '') { const el = $('#sim-state'); el.textContent = label; el.className = `badge ${cls}`; }
function clearOverlay() { ctx.clearRect(0, 0, overlay.width, overlay.height); }
function paintBoxes(detections, laneEstimate = {}) {
  const rect = video.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  overlay.width = Math.max(1, Math.round(rect.width * dpr)); overlay.height = Math.max(1, Math.round(rect.height * dpr));
  overlay.style.width = `${rect.width}px`; overlay.style.height = `${rect.height}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); clearOverlay();
  if (!video.videoWidth || !video.videoHeight) return;
  const scale = Math.min(rect.width / video.videoWidth, rect.height / video.videoHeight);
  const ox = (rect.width - video.videoWidth * scale) / 2, oy = (rect.height - video.videoHeight * scale) / 2;
  ctx.lineWidth = 2; ctx.font = '600 12px Manrope, sans-serif';
  for (const d of detections) {
    const [x1, y1, x2, y2] = d.bbox, x = ox + x1 * scale, y = oy + y1 * scale, w = (x2-x1)*scale, h = (y2-y1)*scale;
    ctx.strokeStyle = '#51e3ad'; ctx.fillStyle = '#51e3ad'; ctx.strokeRect(x,y,w,h);
    const label = `${d.label} ${(d.confidence*100).toFixed(0)}%`;
    ctx.fillRect(x, Math.max(0,y-20), ctx.measureText(label).width+12, 20); ctx.fillStyle = '#08110e'; ctx.fillText(label,x+6,Math.max(14,y-6));
  }
  for (const lane of (laneEstimate?.lanes || [])) {
    if(lane.points?.length===2){const [[x1,y1],[x2,y2]]=lane.points;ctx.strokeStyle='#ffd166';ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(ox+x1*scale,oy+y1*scale);ctx.lineTo(ox+x2*scale,oy+y2*scale);ctx.stroke();}
  }
}
function renderSimMetrics(latency = null) {
  const total = sim.detections.length, classes = Object.entries(sim.classes).map(([k,v])=>`${esc(k)} ${v}`).join(' · ') || 'No detections';
  const average = sim.samples ? sim.inferenceMs / sim.samples : null;
  const fps = latency > 0 ? (1000 / latency).toFixed(2) : '—';
  $('#sim-metrics').innerHTML = `<div><small>FRAMES PRESENTED</small><b>${sim.presented}</b></div><div><small>FRAMES SAMPLED</small><b>${sim.samples}</b></div><div><small>SAMPLES SKIPPED WHILE BUSY</small><b>${sim.skippedSamples}</b></div><div><small>VEHICLE DETECTION INSTANCES</small><b>${sim.vehicleInstances}</b></div><div><small>POTHOLE CANDIDATE EVENTS</small><b>${sim.potholeEvents}</b></div><div><small>DRIVER INDICATOR EVENTS</small><b>${sim.driverAlerts}</b></div><div><small>LANE ESTIMATES</small><b>${sim.laneAvailableSamples}/${sim.samples}</b></div><div><small>EXPERIMENTAL RISK ALERTS</small><b>${sim.riskAlerts}</b></div><div><small>ALL DETECTION INSTANCES · FRAME SCOPE</small><b>${total} · ${classes}</b></div><div><small>LAST PIPELINE LATENCY</small><b>${latency === null ? '—' : `${latency.toFixed(1)} ms`}</b></div><div><small>AVERAGE PIPELINE LATENCY</small><b>${average === null ? '—' : `${average.toFixed(1)} ms`}</b></div><div><small>LAST PIPELINE RATE</small><b>${fps} fps</b></div><div><small>ELAPSED SIMULATION</small><b>${fmtTime(performance.now()/1000-sim.startedAt/1000)}</b></div>`;
}
function setCapability(id, title, detail) {
  const el = $(`#cap-${id}`); if (!el) return;
  const status = String(title).toUpperCase();
  el.className = `capability ${/UNAVAILABLE|ERROR|NOT_VISIBLE|MISSING_MODEL|NOT_IMPLEMENTED/.test(status) ? 'unavailable' : /READY|ESTIMATED|VISIBLE|INFERENCE_COMPLETE/.test(status) ? 'observed' : ''}`;
  el.querySelector('span').textContent = detail;
}
function showRisk(risk) {
  if (!risk) return;
  const box = $('#sim-alert');
  if (risk.alert) {
    const evidence = risk.alert.evidence?.join('; ');
    box.className = 'sim-alert warning'; box.textContent = `${risk.alert.risk_level || 'EXPERIMENTAL'} · Potential forward obstacle @ ${fmtTime(risk.timestamp)}${evidence ? ` · Evidence: ${evidence}` : ''} · ${risk.uncertainty || 'Requires human review'}`;
  } else {
    box.className = 'sim-alert'; box.textContent = `${risk.status || 'UNAVAILABLE'} · ${risk.message || risk.reason || 'Experimental assessment only; no crash prediction.'} ${risk.uncertainty || ''}`;
  }
}
function updateProgress() {
  const t = video.currentTime || 0, d = video.duration || 0;
  $('#sim-time').textContent = `${fmtTime(t)} / ${fmtTime(d)}`;
  if (!$('#sim-seek').matches(':active')) $('#sim-seek').value = d ? Math.round(1000*t/d) : 0;
}
$('#sim-file').onchange = event => {
  const file = event.target.files[0]; if (!file) return;
  if (sim.active || sim.starting || sim.busy) { $('#sim-message').textContent='Stop the active session before selecting another video.'; event.target.value=''; return; }
  if (file.size > MAX_UPLOAD_BYTES) { $('#sim-message').textContent = 'Video exceeds the 250 MB limit.'; event.target.value=''; return; }
  if (!file.type.startsWith('video/') && !/\.(mp4|mov|avi|mkv|webm)$/i.test(file.name)) { $('#sim-message').textContent='Choose a supported video file.'; return; }
  sim.file = file; if (sim.url) URL.revokeObjectURL(sim.url); sim.url = URL.createObjectURL(file);
  video.src = sim.url; video.load(); sim.active=false; sim.sessionId=null; sim.completed=false;
  $('#sim-start').disabled=false; $('#sim-pause').disabled=$('#sim-resume').disabled=$('#sim-restart').disabled=$('#sim-stop').disabled=true;
  $('#sim-seek').disabled=true; $('#sim-message').textContent='Local preview ready. The backend model will load when you start.';
  $('#sim-overlay-status').textContent='Local prerecorded video · model inference not started'; setSimState('Ready');
};
video.onloadedmetadata = () => { $('#sim-seek').disabled=false; updateProgress(); const w=video.videoWidth,h=video.videoHeight; $('#video-meta').textContent=`Decoded video · ${w} × ${h} · ${w>=h?'landscape':'portrait'} · ${video.duration.toFixed(2)} s · aspect ratio preserved · inference uses max ${MAX_INFERENCE_FRAME_EDGE} px edge`; if(video.duration>180){$('#sim-start').disabled=true;$('#sim-message').textContent='Videos longer than 180 seconds are not supported.';} else if(sim.file){$('#sim-start').disabled=false;} };
video.ontimeupdate = updateProgress;
video.onerror = () => { if (sim.active) finishSim('error', 'Video could not be decoded by this browser.'); else $('#sim-message').textContent='This browser cannot decode the selected video format.'; };
video.onended = () => { if (sim.active && !sim.completed) finishSim('complete', 'Video replay completed.'); };
$('#sim-speed').onchange = () => { video.playbackRate = Number($('#sim-speed').value); };
$('#sim-seek').oninput = () => { if (video.duration) video.currentTime = Number($('#sim-seek').value)/1000*video.duration; };
async function startSim(restart = false) {
  if (!sim.file || sim.busy || sim.starting) return;
  sim.starting=true; $('#sim-start').disabled=true; const profile=$('#sim-profile').value; $('#sim-message').textContent=`Initializing modules for ${profile.replaceAll('_',' ')} video…`;
  if (sim.sessionId) { try { await api(`/api/simulation/${sim.sessionId}/discard`, {method:'POST'}); } catch {} }
  sim.active=false; sim.sessionId=null; sim.detections=[]; sim.classes={}; sim.alerts=[]; sim.lastSampleTime=-1; sim.frameId=0; sim.presented=0; sim.samples=0; sim.skippedSamples=0; sim.inferenceMs=0; sim.vehicleInstances=0; sim.potholeEvents=0; sim.driverAlerts=0; sim.laneAvailableSamples=0; sim.riskAlerts=0; sim.lastRiskTimestamp=-Infinity; sim.completed=false; sim.pendingFinish=null;
  sim.lastHazardTimestamp=undefined;
  if (restart || video.ended || video.currentTime >= (video.duration || Infinity)) video.currentTime=0;
  clearOverlay(); renderSimMetrics(); $('#sim-timeline').innerHTML=''; $('#sim-message').textContent='Loading selected detector and starting session…';
  try {
    const confidence=Number($('#sim-confidence').value);
    if (!Number.isFinite(confidence)||confidence<.05||confidence>.95) throw Error('Confidence must be between 0.05 and 0.95.');
    const started=await api('/api/simulation/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:sim.file.name,detector:'vehicle',profile,confidence,sampling_rate_hz:Number($('#sim-rate').value),features:['drowsiness','lane','collision_risk']})});
    sim.sessionId=started.session_id; sim.active=true; sim.paused=false; sim.startedAt=performance.now();
    $('#sim-seek').disabled=true;
    $('#sim-model').textContent=`${started.model_used} · ${started.confidence_threshold.toFixed(2)} · ${started.sampling_rate_hz} Hz`;
    $('#profile-assessment').textContent=`${started.assessment.message} Selected profile: ${started.profile.replaceAll('_',' ')}.`;
    setCapability('vehicle',started.capability_status.vehicle.toUpperCase(),started.capability_status.vehicle==='not_applicable'?'Not selected for this video profile':started.capability_status.vehicle==='ready'?'YOLO inference already verified in this service process':'YOLO model initialized; first inference pending');
    setCapability('pothole',started.capability_status.pothole,started.capability_status.pothole==='unavailable'?'Pothole model or inference dependency unavailable':started.capability_status.pothole==='not_applicable'?'Not selected for this video profile':'Experimental segmentation attempted on each sampled frame');
    setCapability('driver',started.capability_status.drowsiness,started.capability_status.drowsiness==='unavailable'?'MediaPipe unavailable; no face-based conclusion':started.capability_status.drowsiness==='not_applicable'?'Not selected for this video profile':'Experimental · checking whether a single usable face is visible');
    setCapability('lane',started.capability_status.lane.toUpperCase(),started.capability_status.lane==='not_applicable'?'Not selected for this video profile':'OpenCV baseline evaluates sampled frames');
    setCapability('risk',started.capability_status.collision_risk.toUpperCase(),started.capability_status.collision_risk==='not_applicable'?'Not selected for this video profile':'Temporal image-plane indicator; not a crash prediction');
    $('#sim-start').disabled=true; $('#sim-pause').disabled=false; $('#sim-resume').disabled=true; $('#sim-restart').disabled=false; $('#sim-stop').disabled=false;
    $('#sim-message').textContent='Running sequential inference on sampled playback frames.'; setSimState('Running','ready');
    $('#sim-overlay-status').textContent='Inference overlays are estimates; inspect the detections below.';
    video.playbackRate=Number($('#sim-speed').value); await video.play();
  } catch(error) { sim.active=false; $('#sim-message').textContent=error.message; setSimState('Unavailable','unavailable'); $('#sim-start').disabled=!sim.file; }
  finally { sim.starting=false; }
}
async function processPresentedFrame(time, metadata) {
  sim.busy=true;
  clearOverlay(); $('#sim-overlay-status').textContent=`Analyzing frame @ ${fmtTime(time)} · playback continues`;
  try {
      const c=document.createElement('canvas');
      const resize=Math.min(1,MAX_INFERENCE_FRAME_EDGE/Math.max(video.videoWidth,video.videoHeight));
      c.width=Math.max(1,Math.round(video.videoWidth*resize)); c.height=Math.max(1,Math.round(video.videoHeight*resize));
      c.getContext('2d').drawImage(video,0,0,c.width,c.height);
      const blob=await new Promise(resolve=>c.toBlob(resolve,'image/jpeg',.86)); if (!blob) throw Error('Could not capture a video frame.');
      const form=new FormData(); form.append('file',blob,'frame.jpg');
      const frameId=Math.max(sim.frameId+1,Number(metadata.presentedFrames)||sim.frameId+1); sim.frameId=frameId;
      const meta=new URLSearchParams({timestamp:time.toFixed(4),frame_id:String(frameId),width:String(c.width),height:String(c.height)});
      const result=await api(`/api/simulation/${sim.sessionId}/frame?${meta}`,{method:'POST',body:form});
      const pa=result.profile_assessment||{}; $('#profile-assessment').textContent=`${pa.profile||'video'} profile · Road cues: ${pa.road_scene||'unknown'} (${pa.road_evidence_frames||0} sampled frame(s)); driver face: ${pa.driver_face||'not assessed'} (${pa.driver_face_visible_samples||0} sample(s)). ${pa.message||''}`;
      const overlayCurrent=video.paused && Math.abs(video.currentTime-time)<=0.1;
      const sx=video.videoWidth/c.width, sy=video.videoHeight/c.height;
      const displayDetections=result.detections.map(d=>({...d,bbox:d.bbox.map((n,i)=>n*(i%2?sy:sx))}));
      const displayLane={...result.lane_estimate,lanes:(result.lane_estimate?.lanes||[]).map(l=>({...l,points:l.points?.map(([x,y])=>[x*sx,y*sy])}))};
      if(overlayCurrent) paintBoxes(displayDetections,displayLane); else clearOverlay();
      sim.samples++; sim.detections.push(...displayDetections);
      for(const d of displayDetections) { sim.classes[d.label]=(sim.classes[d.label]||0)+1; if(d.detector_id==='vehicle') sim.vehicleInstances++; }
      sim.inferenceMs+=result.processing_latency_ms; showRisk(result.risk_assessment); renderSimMetrics(result.processing_latency_ms);
      const resultText=`${result.detections.length} detection${result.detections.length===1?'':'s'} · ${result.processing_latency_ms.toFixed(0)} ms · ${fmtTime(time)}`;
      $('#sim-overlay-status').textContent=overlayCurrent?resultText:`${result.inference_status==='delayed'?'INFERENCE DELAYED · ':''}Results @ ${fmtTime(time)} · ${resultText.split(' · ').slice(0,2).join(' · ')} · playback continues; stale overlay hidden`;
      const driver=result.driver_monitoring||{};
      setCapability('driver',driver.status||'UNAVAILABLE',driver.status==='FACE_NOT_VISIBLE'?'Driver not visible — monitoring unavailable for this footage':driver.indicator||driver.status||'Insufficient evidence');
      setCapability('vehicle',result.module_status?.vehicle||'READY',result.module_status?.vehicle==='NOT_APPLICABLE'?'Not run for driver-facing profile':'YOLO vehicle inference completed on sampled frames');
      setCapability('lane',result.lane_estimate?.status||'UNAVAILABLE',result.lane_estimate?.status==='ESTIMATED'?'Lane boundaries estimated in image space':result.lane_estimate?.reason||'Lane estimate unavailable for this frame');
      if(result.lane_estimate?.status==='ESTIMATED') sim.laneAvailableSamples++;
      setCapability('pothole',result.pothole_status?.status||'UNAVAILABLE',result.pothole_status?.status==='INFERENCE_COMPLETE'?`${result.pothole_status.detections?.length||0} experimental segmentation detection(s) on this sample`:result.pothole_status?.message||'Pothole inference unavailable');
      if(result.module_status?.collision_risk==='UNAVAILABLE') setCapability('risk','UNAVAILABLE',result.risk_assessment?.reason||'Insufficient temporal evidence');
      if(driver.indicator==='POSSIBLE_PROLONGED_EYE_CLOSURE') { sim.driverAlerts++; $('#sim-timeline').insertAdjacentHTML('afterbegin',`<div class="timeline-alert">DRIVER MONITOR · Possible prolonged eye closure @ ${fmtTime(time)} · Experimental indicator</div>`); }
      const hazardEvent=result.pothole_status?.detections?.length && (sim.lastHazardTimestamp===undefined || time-sim.lastHazardTimestamp>=3);
      if(hazardEvent) { sim.lastHazardTimestamp=time; sim.potholeEvents++; $('#sim-timeline').insertAdjacentHTML('afterbegin',`<div class="timeline-alert">POTHOLE MODEL · ${result.pothole_status.detections.length} candidate(s) @ ${fmtTime(time)} · Experimental, review for false positives</div>`); }
      if(result.risk_assessment?.alert && time-sim.lastRiskTimestamp>=5) {
        sim.lastRiskTimestamp=time; sim.riskAlerts++;
        sim.alerts.push(result.risk_assessment);
        setCapability('risk',result.risk_assessment.alert.risk_level||'EXPERIMENTAL WARNING',`Evidence observed @ ${fmtTime(time)} · ${result.risk_assessment.uncertainty||'Not a prediction'}`);
        $('#sim-timeline').insertAdjacentHTML('afterbegin',`<div class="timeline-alert"><b>${esc(result.risk_assessment.level||'EXPERIMENTAL ALERT')}</b> · ${fmtTime(time)} · ${esc(result.risk_assessment.message||'Potential forward obstacle')}</div>`);
      } else if(result.module_status?.collision_risk==='UNAVAILABLE') setCapability('risk','UNAVAILABLE',result.risk_assessment?.reason||'Insufficient temporal evidence; not evidence of a clear road');
  } catch(error) { await finishSim('error',error.message); }
  finally { sim.busy=false; }
  if(sim.pendingFinish){const pending=sim.pendingFinish;sim.pendingFinish=null;await finishSim(pending.status,pending.message);}
}
function onPresentedFrame(now, metadata) {
  sim.callback=null;
  if (!sim.active) return;
  sim.presented=Math.max(sim.presented, Number(metadata.presentedFrames)||0);
  const time=video.currentTime, interval=1/Number($('#sim-rate').value), elapsed=time-sim.lastSampleTime;
  if (elapsed >= interval-0.004 && video.videoWidth) {
    if (!sim.busy) {
      sim.lastSampleTime=time;
      void processPresentedFrame(time, metadata);
    } else {
      const skipped=Math.max(1,Math.floor((elapsed+0.004)/interval));
      sim.skippedSamples+=skipped;
      sim.lastSampleTime+=skipped*interval;
      renderSimMetrics();
    }
  } else if (!sim.busy && sim.lastSampleTime >= 0 && time > sim.lastSampleTime) {
    clearOverlay(); $('#sim-overlay-status').textContent=`Playback · next inference near ${fmtTime(sim.lastSampleTime+interval)}`;
  }
  if(sim.active && !video.paused && 'requestVideoFrameCallback' in video) sim.callback=video.requestVideoFrameCallback(onPresentedFrame);
}
video.onplay = () => { sim.paused=false; if(sim.active && 'requestVideoFrameCallback' in video && sim.callback===null) sim.callback=video.requestVideoFrameCallback(onPresentedFrame); };
video.onpause = () => { if(sim.active && !video.ended) sim.paused=true; if(sim.callback!==null && 'cancelVideoFrameCallback' in video) video.cancelVideoFrameCallback(sim.callback); sim.callback=null; };
$('#sim-start').onclick=()=>startSim(false);
$('#sim-restart').onclick=()=>startSim(true);
$('#sim-pause').onclick=()=>{video.pause();sim.paused=true;$('#sim-pause').disabled=true;$('#sim-resume').disabled=false;setSimState('Paused');};
$('#sim-resume').onclick=async()=>{if(!sim.active)return;sim.paused=false;$('#sim-pause').disabled=false;$('#sim-resume').disabled=true;setSimState('Running','ready');try{await video.play();}catch(e){await finishSim('error',e.message);}};
async function finishSim(status,message) {
  if(!sim.active||sim.completed)return; video.pause();
  if(sim.busy){sim.pendingFinish={status,message};sim.paused=true;return;}
  sim.completed=true; sim.active=false;
  $('#sim-pause').disabled=$('#sim-resume').disabled=$('#sim-stop').disabled=true; $('#sim-start').disabled=false;
  $('#sim-seek').disabled=false;
  let saved=null;
  try { if(sim.sessionId) saved=await api(`/api/simulation/${sim.sessionId}/finish`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({frames_presented:sim.presented,status})}); }
  catch(error) { message=`${message} Save failed: ${error.message}`; }
  sim.sessionId=null; setSimState(status==='complete'?'Complete':status==='stopped'?'Stopped':'Error',status==='complete'?'ready':'unavailable');
  $('#sim-message').textContent=`${message}${saved?.persisted?` Persisted ${saved.total_detections} detections across ${saved.frames_sampled} sampled frames; average inference ${(saved.average_inference_ms||0).toFixed(1)} ms.`:''}`;
  $('#sim-overlay-status').textContent=`${status==='complete'?'Replay complete':status==='stopped'?'Replay stopped':'Replay ended with an error'} · ${sim.samples} sampled frames`;
  if(saved?.persisted) { await Promise.all([refreshStats(),loadEvents()]); }
}
$('#sim-stop').onclick=()=>finishSim('stopped','Replay stopped by user.');
async function refreshAll() {
  try { await Promise.all([refreshStats(), loadModels(), loadEvents()]); }
  catch (error) { $('.connection span').textContent = 'API OFFLINE'; $('#model-list').innerHTML = `<div class="empty">API unavailable: ${esc(error.message)}</div>`; }
}
refreshAll();
