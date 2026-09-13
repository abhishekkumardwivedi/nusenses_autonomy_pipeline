const video = document.getElementById('video');
const status = document.getElementById('status');
const stats = document.getElementById('stats');
const startButton = document.getElementById('start');
const stopButton = document.getElementById('stop');
let pc = null, peerId = null, interval = null, deadline = null, controller = null;

function stopStream(message = 'Stream stopped.') {
  const id = peerId;
  peerId = null;
  controller?.abort();
  controller = null;
  clearInterval(interval);
  clearTimeout(deadline);
  const old = pc;
  pc = null;
  old?.close();
  video.srcObject?.getTracks().forEach(track => track.stop());
  video.srcObject = null;
  if (id) fetch(`/stop/${id}`, {method: 'POST', keepalive: true}).catch(console.warn);
  status.textContent = 'Disconnected';
  stats.textContent = message;
  startButton.disabled = false;
  stopButton.disabled = true;
}

function waitForIce(peer) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => finish(new Error('ICE gathering timed out')), 20000);
    function finish(error) {
      clearTimeout(timeout);
      peer.removeEventListener('icegatheringstatechange', check);
      error ? reject(error) : resolve();
    }
    function check() { if (peer.iceGatheringState === 'complete') finish(); }
    peer.addEventListener('icegatheringstatechange', check);
    check();
  });
}

async function startStream() {
  if (pc) return;
  startButton.disabled = true;
  stopButton.disabled = false;
  status.textContent = 'Connecting';
  stats.textContent = 'Negotiating WebRTC…';
  const peer = new RTCPeerConnection({iceServers});
  pc = peer;
  controller = new AbortController();
  const signal = controller.signal;
  let previous = null;
  deadline = setTimeout(() => {
    if (pc === peer) stopStream('Connection timed out. Check ICE/TURN connectivity and server logs.');
  }, 45000);
  peer.ontrack = event => {
    if (pc !== peer) return;
    video.srcObject = event.streams[0] || new MediaStream([event.track]);
    video.play().catch(console.warn);
  };
  peer.oniceconnectionstatechange = () => console.log('ICE:', peer.iceConnectionState);
  peer.onconnectionstatechange = () => {
    console.log('Connection:', peer.connectionState);
    if (pc !== peer) return;
    if (peer.connectionState === 'connected') {
      clearTimeout(deadline);
      status.textContent = 'Connected';
    } else if (['failed', 'closed', 'disconnected'].includes(peer.connectionState)) {
      stopStream('Peer disconnected. If ICE failed, check TURN/network connectivity.');
    }
  };
  try {
    peer.addTransceiver('video', {direction: 'recvonly'});
    await peer.setLocalDescription(await peer.createOffer());
    await waitForIce(peer);
    if (pc !== peer) return;
    const response = await fetch('/offer', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(peer.localDescription), signal
    });
    if (!response.ok) throw new Error(await response.text());
    const answer = await response.json();
    if (pc !== peer) {
      fetch(`/stop/${answer.peer_id}`, {method: 'POST', keepalive: true}).catch(console.warn);
      return;
    }
    peerId = answer.peer_id;
    await peer.setRemoteDescription({type: answer.type, sdp: answer.sdp});
    interval = setInterval(async () => {
      try {
        const reports = await peer.getStats();
        if (pc !== peer) return;
        reports.forEach(report => {
          if (report.type !== 'inbound-rtp' || report.kind !== 'video') return;
          const frames = report.framesDecoded || 0;
          const seconds = previous ? (report.timestamp - previous.timestamp) / 1000 : 0;
          const fps = seconds > 0 ? (frames - previous.frames) / seconds : 0;
          stats.textContent = `Received FPS: ${fps.toFixed(1)} | Decoded frames: ${frames} | Bytes: ${report.bytesReceived} | ${video.videoWidth} × ${video.videoHeight}`;
          previous = {timestamp: report.timestamp, frames};
        });
      } catch (error) { console.warn(error); }
    }, 1000);
  } catch (error) {
    console.error(error);
    if (pc === peer) stopStream(`Error: ${error.message}`);
  }
};
startButton.onclick = startStream;
stopButton.onclick = () => stopStream();
window.addEventListener('pagehide', () => stopStream());
