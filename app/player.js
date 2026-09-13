const sceneSelect = document.getElementById('scene');
const rateSelect = document.getElementById('rate');
const playbackStatus = document.getElementById('sample-status');
const errorText = document.getElementById('error');
let busy = false, controlError = '';
let latestState = null, lastLayout = '';
const focusSelect = document.getElementById('focus');
const featureCamera = document.getElementById('feature-camera');
const featureMode = document.getElementById('feature-mode');
const featureChannel = document.getElementById('feature-channel');
const cameras = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT'];
for (const name of [...cameras, 'BEV', 'LIDAR', 'RADAR', 'FEATURE', 'SPATIAL']) focusSelect.add(new Option(name === 'SPATIAL' ? 'Camera Spatial BEV' : name, name));
for (const name of cameras) featureCamera.add(new Option(name, name));
featureCamera.value = 'CAM_FRONT';
function updateInspection(state) {
  latestState = state;
  focusSelect.value = state.view.focus;
  featureCamera.value = state.view.feature_camera;
  featureMode.value = state.view.feature_mode;
  if (document.activeElement !== featureChannel) featureChannel.value = state.view.feature_channel;
  featureChannel.disabled = state.view.feature_mode !== 'channel' || !state.encoder.enabled;
  featureCamera.disabled = featureMode.disabled = !state.encoder.enabled;
  focusSelect.querySelector('option[value="FEATURE"]').disabled = !state.encoder.enabled;
  focusSelect.querySelector('option[value="SPATIAL"]').disabled = !state.spatial;
  document.getElementById('heading').textContent = `nuScenes Stage ${state.stage} · ${state.encoder.enabled ? 'Camera Encoder + ' : ''}Multi-Sensor Playback`;
  document.getElementById('stage-label').textContent = `Recorded real-world sensor data · AI inference: Camera Encoder ${state.encoder.enabled ? 'ENABLED' : 'DISABLED (Stage 2)'}`;
  if (state.spatial) {
    document.getElementById('heading').textContent = 'Stage 4 — Camera Features → Spatial BEV';
    document.getElementById('stage-label').textContent = 'Camera Spatial BEV · LiDAR-assisted depth for geometry validation · Current sample only; no temporal memory';
  }
  const enc = state.encoder;
  document.getElementById('stage3-steps').hidden = !enc.enabled || !!state.spatial;
  document.getElementById('spatial-note').hidden = !state.spatial;
  const rows = state.spatial ? [
    `CAMERA → SPATIAL BEV\nInput [1,6,3,256,448]\nFeatures [1,6,256,8,14]\nBEV ${JSON.stringify(state.spatial.shape)}`,
    `GEOMETRY\nDepth: nuScenes LiDAR (validation oracle)\nRange ±50 m / resolution 0.5 m\n+X forward / +Y left / +Z up\n${state.spatial.occupied_cells} cells with evidence; gray = unknown`,
    `CURRENT SAMPLE\nEncoder ${state.camera_encoder_ms} ms\nCamera-to-BEV ${state.spatial.camera_to_bev_ms} ms\nTotal processing ${state.total_ms} ms\nGPU allocated ${state.spatial.gpu_allocated_mb} MiB`
  ] : enc.enabled ? [
    `MODEL / WEIGHTS\n${enc.model} / ${enc.weights}\n${enc.projection}`,
    `DEVICE / DTYPE\n${enc.device} / ${enc.gpu}\n${enc.dtype}; input ${enc.input_dtype}\nCache ${enc.cache_device} / ${enc.cache_dtype}`,
    `TENSORS\nInput ${JSON.stringify(enc.input_shape)}\nBackbone batch ${JSON.stringify(enc.backbone_batch_shape)}\nBackbone output ${JSON.stringify(enc.backbone_output_shape)}\nOutput ${JSON.stringify(enc.output_shape)}`,
    `TIMING (last sample)\nLoad ${state.sensor_load_ms} ms\nPreprocess ${state.preprocess_ms} ms\nEncoder ${state.camera_encoder_ms} ms\nRender ${state.render_ms} ms\nTotal ${state.total_ms} ms\nLast view-only render ${state.view_render_ms} ms`,
    `GPU MEMORY (MiB)\nAllocated ${enc.gpu_allocated_mb}\nReserved ${enc.gpu_reserved_mb}\nPeak allocated ${enc.gpu_peak_mb}`,
    `CACHE / RUNTIME\nInference calls ${enc.inference_count}\nGrid ${enc.feature_grid.join(' × ')} / ${enc.feature_channels} channels\nPyTorch ${enc.torch}\ntorchvision ${enc.torchvision}\nCUDA ${enc.cuda}`
  ] : ['Stage 2: neural inference disabled. Sensor loading, transforms, playback and WebRTC remain active.'];
  document.getElementById('runtime').replaceChildren(...rows.map(label => {
    const p = document.createElement('p'); p.textContent = label; return p;
  }));
  const layout = JSON.stringify([state.view.focus, state.panels]);
  if (layout !== lastLayout) {
    lastLayout = layout;
    const buttons = document.getElementById('expand-buttons');
    buttons.replaceChildren();
    const panels = state.view.focus === 'overview' ? state.panels : {overview:[0,0,1600,900]};
    for (const [name, rect] of Object.entries(panels)) {
      const button = document.createElement('button');
      button.className = 'expand';
      button.textContent = name === 'overview' ? '↙ Overview' : `↗ ${name === 'SPATIAL' ? 'Camera BEV' : name}`;
      button.setAttribute('aria-label', name === 'overview' ? 'Minimize panel' : `Expand ${name}`);
      button.style.left = `${(rect[0] + rect[2] - 8) / 16}%`;
      button.style.top = `${(rect[1] + 6) / 9}%`;
      button.style.transform = 'translateX(-100%)';
      button.onclick = () => command('focus', name);
      buttons.append(button);
    }
  }
}
function showState(state) {
  sceneSelect.value = state.scene_token;
  rateSelect.value = String(state.rate);
  playbackStatus.textContent = `${state.scene} | Sample ${state.sample_index}/${state.total_samples} | Timestamp ${state.timestamp} µs | ${state.playing ? 'Playing' : 'Paused'} | Source ${state.source_fps} Hz | Playback ${state.playback_fps} samples/s | LiDAR ${state.lidar_points} · Radar ${state.radar_points} | Load+render ${state.load_render_ms} ms`;
  errorText.textContent = state.error || controlError;
  document.getElementById('details').textContent = JSON.stringify({scene_token:state.scene_token, sample_token:state.sample_token, timestamp:state.timestamp, ego_pose:state.ego_pose, sensors:state.sensors}, null, 2);
  updateInspection(state);
}
async function command(action, value) {
  if (busy) return;
  busy = true;
  try {
    const response = await fetch('/control', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({action,value})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    controlError = '';
    showState(data);
  } catch (error) { controlError = error.message; errorText.textContent = controlError; }
  finally { busy = false; }
}
document.getElementById('play').onclick = async () => {
  if (!pc) await startStream();
  if (pc) await command('play');
};
for (const action of ['pause','previous','next','restart']) {
  document.getElementById(action).onclick = () => command(action);
}
sceneSelect.onchange = () => command('scene', sceneSelect.value);
rateSelect.onchange = () => command('rate', rateSelect.value);
focusSelect.onchange = () => command('focus', focusSelect.value);
featureCamera.onchange = () => command('feature_camera', featureCamera.value);
featureMode.onchange = () => command('feature_mode', featureMode.value);
featureChannel.onchange = () => {
  if (!featureChannel.checkValidity() || featureChannel.value === '') {
    controlError = 'Feature channel must be an integer from 0 to 255'; errorText.textContent = controlError; return;
  }
  command('feature_channel', Number(featureChannel.value));
};
document.getElementById('overview').onclick = () => command('focus', 'overview');
window.addEventListener('keydown', event => { if (event.key === 'Escape') command('focus', 'overview'); });
video.ondblclick = event => {
  if (!latestState) return;
  if (latestState.view.focus !== 'overview') { command('focus','overview'); return; }
  const box = video.getBoundingClientRect();
  const [w,h] = latestState.video_size;
  const scale = Math.min(box.width/w, box.height/h);
  const x = (event.clientX - box.left - (box.width-w*scale)/2)/scale;
  const y = (event.clientY - box.top - (box.height-h*scale)/2)/scale;
  for (const [name,[px,py,pw,ph]] of Object.entries(latestState.panels)) {
    if (x >= px && x < px+pw && y >= py && y < py+ph) { command('focus',name); break; }
  }
};
async function refresh() {
  try {
    if (!busy) {
      const response = await fetch('/state', {cache:'no-store'});
      if (!response.ok) throw new Error(`State HTTP ${response.status}`);
      const data = await response.json();
      if (!busy) showState(data);
    }
  } catch (error) { errorText.textContent = error.message; }
  setTimeout(refresh, 500);
}
(async () => {
  try {
    const response = await fetch('/scenes');
    if (!response.ok) throw new Error(`Scenes HTTP ${response.status}`);
    for (const scene of await response.json()) {
      sceneSelect.add(new Option(`${scene.name} (${scene.samples} samples)`, scene.token));
    }
    const selected = new URLSearchParams(location.search).get('scene');
    if (selected) await command('scene', selected);
    sceneSelect.disabled = false;
    document.querySelectorAll('[data-control]').forEach(button => button.disabled = false);
    refresh();
  } catch (error) { errorText.textContent = error.message; }
})();
