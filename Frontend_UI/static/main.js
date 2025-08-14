(() => {
	"use strict";

	const qs = (sel) => document.querySelector(sel);
	const qsa = (sel) => Array.from(document.querySelectorAll(sel));

	const schemaInput = qs('#schemaInput');
	const saveSchemaBtn = qs('#saveSchema');
	const schemaStatus = qs('#schemaStatus');
	const schemaTemplateSelect = qs('#schemaTemplateSelect');
	const insertSchemaTemplateBtn = qs('#insertSchemaTemplate');
	const clearSchemaBtn = qs('#clearSchema');

	const dropZone = qs('#dropZone');
	const fileInput = qs('#fileInput');
	const fileMeta = qs('#fileMeta');
	const fileTitle = qs('#fileTitle');

	const columnsDiv = qs('#columns');
	const filtersInput = qs('#filters');
	const tableContainer = qs('#tableContainer');

	const consoleEl = qs('#console');
	const consolePanel = qs('#consolePanel');
	const modalExtractSchemaBtn = qs('#modalExtractSchema');
	const modalGenerateEmbeddingsBtn = qs('#modalGenerateEmbeddings');

	const graphPanel = qs('#graphPanel');
	const graphPlot = qs('#graphPlot');
	const modalFilesNav = qs('#modalFiles');
	const scanModalBtn = qs('#scanModalBtn');
	const scanStatus = qs('#scanStatus');
	const buildGraphBtn = qs('#buildGraphBtn');
	const rowsInput = qs('#rowsInput');
	const simpleInfo = qs('#simpleInfo');
	const buildCharsBtn = qs('#buildCharsBtn');
	const cleanViewBtn = qs('#cleanViewBtn');
	const storyBtn = qs('#storyBtn');
	const stopBtn = qs('#stopBtn');

	let selectedModalFile = null;
	let modalFileTotalRows = 0;

	let currentColumns = [];
	let selectedColumns = new Set();
	let expandedColumns = new Set();

	// Graph state and helpers
	const graphState = {
		nodesById: new Map(),
		edges: [],
		resizeHandler: null,
		rotationTimer: null,
		angle: 0,
		hasPlot: false,
		highlightId: null,
		cameraBound: false,
	};

	function getSavedCamera(){
		try{ const s = localStorage.getItem('graphCamera'); if(!s) return null; return JSON.parse(s); }catch(e){ return null; }
	}

	function setSavedCamera(cam){
		try{ localStorage.setItem('graphCamera', JSON.stringify(cam)); }catch(e){ /* ignore */ }
	}

	function bindRelayoutSave(){
		if(graphState.cameraBound || !graphPlot) return;
		graphPlot.on('plotly_relayout', (relayoutData)=>{
			let cam = getSavedCamera() || { eye:{x:1.6,y:1.6,z:0.8}, center:{x:0,y:0,z:0} };
			if(relayoutData['scene.camera']){
				cam = relayoutData['scene.camera'];
			} else {
				for(const k in relayoutData){
					const v = relayoutData[k];
					if(k.startsWith('scene.camera.')){
						const path = k.split('.'); // [scene,camera,eye,x]
						if(path.length===4){ const [, , part, axis] = path; cam[part] = cam[part]||{}; cam[part][axis] = v; }
					}
				}
			}
			setSavedCamera(cam);
		});
		graphState.cameraBound = true;
	}

	function resetGraph(){
		if(graphState.resizeHandler){
			window.removeEventListener('resize', graphState.resizeHandler);
			graphState.resizeHandler = null;
		}
		if(graphState.rotationTimer){
			clearInterval(graphState.rotationTimer);
			graphState.rotationTimer = null;
		}
		try{ if(graphPlot) Plotly.purge(graphPlot); }catch(e){}
		graphState.nodesById.clear();
		graphState.edges = [];
		graphState.hasPlot = false;
		graphState.highlightId = null;
	}

	function hashString(str){
		let h = 2166136261 >>> 0;
		for(let i=0;i<str.length;i++){
			h ^= str.charCodeAt(i);
			h = Math.imul(h, 16777619);
		}
		return h >>> 0;
	}

	function positionForNode(id){
		const h = hashString(String(id));
		const theta = (h % 360) * Math.PI / 180;
		const phi = ((h >>> 9) % 180) * Math.PI / 180;
		const r = 1.0;
		const x = r * Math.sin(phi) * Math.cos(theta);
		const y = r * Math.sin(phi) * Math.sin(theta);
		const z = r * Math.cos(phi);
		return {x, y, z};
	}

	function computeDegrees(edges){
		const deg = new Map();
		for(const e of edges){
			deg.set(e.source, (deg.get(e.source)||0)+1);
			deg.set(e.target, (deg.get(e.target)||0)+1);
		}
		return deg;
	}

	function renderPlot(){
		if(!graphPlot) return;
		const nodes = Array.from(graphState.nodesById.values());
		const edges = graphState.edges || [];
		const deg = computeDegrees(edges);
		let centralId = null; let maxDeg = -1;
		for(const [id, d] of deg.entries()) if(d>maxDeg){ centralId = id; maxDeg = d; }
		const center = centralId && graphState.nodesById.get(centralId) ? positionForNode(centralId) : {x:0,y:0,z:0};

		const ex = [], ey = [], ez = [];
		for(const e of edges){
			if(!graphState.nodesById.has(e.source) || !graphState.nodesById.has(e.target)) continue;
			const a = positionForNode(e.source);
			const b = positionForNode(e.target);
			ex.push(a.x, b.x, null);
			ey.push(a.y, b.y, null);
			ez.push(a.z, b.z, null);
		}
		const edgeTrace = {
			x: ex, y: ey, z: ez,
			mode: 'lines', type: 'scatter3d',
			line: {color: 'rgba(200,200,200,0.35)', width: 2},
			hoverinfo: 'skip',
			name: 'edges'
		};

		const nx = []; const ny = []; const nz = []; const labels = []; const colors = [];
		for(const n of nodes){
			const p = positionForNode(n.id);
			nx.push(p.x); ny.push(p.y); nz.push(p.z);
			labels.push(n.label || String(n.id));
			let c = (n.group===1? '#00ffc6' : (n.group===2? '#7aa2f7' : '#ff79c6'));
			if(graphState.highlightId && n.id === graphState.highlightId){ c = '#ffff66'; }
			colors.push(c);
		}
		const nodeTrace = {
			x: nx, y: ny, z: nz,
			mode: 'markers+text', type: 'scatter3d',
			marker: {size: 5, color: colors, line: {color: 'white', width: 0.5}},
			text: labels, textposition: 'top center',
			textfont: {size: 14, color: '#c9d1d9'},
			hoverinfo: 'text',
			name: 'nodes'
		};

		const savedCam = getSavedCamera();
		const layout = {
			showlegend: false,
			scene: {
				bgcolor: '#000',
				xaxis: {visible:false}, yaxis: {visible:false}, zaxis: {visible:false},
				camera: savedCam || { eye: { x: 1.6, y: 1.6, z: 0.8 }, center: { x: 0, y: 0, z: 0 } },
				aspectmode: 'cube'
			},
			paper_bgcolor: '#000',
			margin: {l:0, r:0, b:0, t:0},
			height: Math.max(200, graphPanel.clientHeight - 40),
			font: { size: 14, color: '#c9d1d9' }
		};

		try{
			if(!graphState.hasPlot){
				Plotly.newPlot(graphPlot, [edgeTrace, nodeTrace], layout, {
					displayModeBar: true,
					displaylogo: false,
					modeBarButtonsToRemove: ['lasso2d','select2d','autoScale2d','toggleSpikelines'],
				});
				graphState.hasPlot = true;
				setTimeout(()=>{ try{ Plotly.Plots.resize(graphPlot); }catch(e){} }, 0);
			} else {
				Plotly.react(graphPlot, [edgeTrace, nodeTrace], layout, {
					displayModeBar: true,
					displaylogo: false,
					modeBarButtonsToRemove: ['lasso2d','select2d','autoScale2d','toggleSpikelines'],
				});
			}
		}catch(e){ /* ignore */ }

		ensureResizeHandler();
		bindRelayoutSave();
	}

	function log(line){
		consoleEl.textContent += `${line}\n`;
		consoleEl.scrollTop = consoleEl.scrollHeight;
	}

	async function postJSON(url, data){
		const res = await fetch(url, {
			method: 'POST',
			headers: {'Content-Type':'application/json'},
			body: JSON.stringify(data)
		});
		return await res.json();
	}

	function renderColumns(cols){
		columnsDiv.innerHTML = '';
		currentColumns = cols;
		selectedColumns = new Set(cols.slice(0, Math.min(5, cols.length)));
		for(const c of cols){
			const chip = document.createElement('button');
			chip.textContent = c;
			chip.className = 'col-chip' + (selectedColumns.has(c) ? ' active' : '');
			chip.onclick = () => {
				if(selectedColumns.has(c)) selectedColumns.delete(c); else selectedColumns.add(c);
				chip.classList.toggle('active');
			};
			columnsDiv.appendChild(chip);
		}
	}

	function renderTable(preview){
		const cols = preview.columns || [];
		const rows = preview.first_rows || [];
		let html = '<table><thead><tr>' + cols.map(c=>`<th>${c}</th>`).join('') + '</tr></thead><tbody>';
		for(const r of rows){
			html += '<tr>' + cols.map((c, i)=>{
				const val = escapeHtml(r[c]);
				const expanded = expandedColumns.has(i) ? ' expanded' : '';
				return `<td data-col-index="${i}"><div class="cell"><div class="cell-content${expanded}" data-col-index="${i}">${val}</div><button class="cell-ellipsis" data-col-index="${i}" title="Expand">...</button></div></td>`;
			}).join('') + '</tr>';
		}
		html += '</tbody></table>';
		tableContainer.innerHTML = html;
		applyOverflowIndicators();
	}

	// Restore expand handler for preview table ellipsis
	if(tableContainer){
		tableContainer.addEventListener('click', (e) => {
			const target = e.target;
			if(!(target instanceof HTMLElement)) return;
			if(target.classList.contains('cell-ellipsis')){
				const idxStr = target.getAttribute('data-col-index');
				if(idxStr == null) return;
				const idx = Number(idxStr);
				if(!Number.isFinite(idx)) return;
				expandedColumns.add(idx);
				qsa(`.cell-content[data-col-index="${idx}"]`).forEach(el => el.classList.add('expanded'));
				qsa(`.cell-ellipsis[data-col-index="${idx}"]`).forEach(el => { el.style.display = 'none'; });
			}
		});
	}

	function applyOverflowIndicators(){
		qsa('.cell').forEach(cell => {
			const content = cell.querySelector('.cell-content');
			const ellipsis = cell.querySelector('.cell-ellipsis');
			if(!content || !ellipsis) return;
			if(content.classList.contains('expanded')){
				ellipsis.style.display = 'none';
				return;
			}
			requestAnimationFrame(() => {
				const overflows = content.scrollHeight > content.clientHeight + 1;
				ellipsis.style.display = overflows ? 'inline-block' : 'none';
			});
		});
	}

	function escapeHtml(v){
		if(v === null || v === undefined) return '';
		return String(v).replace(/[&<>"']/g, s=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[s]));
	}

	// Schema saving
	saveSchemaBtn.onclick = async () => {
		const schema = schemaInput.value || '';
		const res = await postJSON('/api/schema', {schema});
		schemaStatus.textContent = res.ok ? `Saved (${res.length} chars)` : 'Save failed';
		log('Schema saved.');
	};

	// Schema templates from notebook models
	const SCHEMA_TEMPLATES = {
		"narrative_full": `from pydantic import BaseModel, Field\nfrom typing import List\n\nclass Action(BaseModel):\n    """\n    Represents a concrete physical action between entities in a narrative text.\n    """\n    # Source entity information\n    source: str = Field(..., description="Name of the entity performing the action")\n    source_type: str = Field(..., description="Category of the source (person, animal, object, location)")\n    source_is_character: bool = Field(..., description="Whether the source is a named character")\n\n    # Target entity information\n    target: str = Field(..., description="Name of the entity receiving the action")\n    target_type: str = Field(..., description="Category of the target (person, animal, object, location)")\n    target_is_character: bool = Field(..., description="Whether the target is a named character")\n\n    # Action details\n    action: str = Field(..., description="The verb or short phrase describing the physical interaction")\n    consequence: str = Field(..., description="The immediate outcome or result of the action")\n\n    # Text evidence\n    text_describing_the_action: str = Field(..., description="Text fragment describing the action exactly as it is written in the text")\n    text_describing_the_consequence: str = Field(..., description="Description of the consequence exactly as it is written in the text")\n\n    # Context information\n    location: str = Field(..., description="location from global to local")\n    temporal_order_id: int = Field(..., description="Sequential identifier for chronological order")\n\nclass NarrativeAnalysis(BaseModel):\n    """\n    Simplified analysis of narrative text with entities as strings and actions indexed by name.\n    """\n    text_id: str = Field(..., description="Unique identifier for the analyzed text segment")\n    text_had_no_actions: bool = Field(\n        default=False,\n        description="Whether the text had no actions to extract"\n    )\n    actions: List[Action] = Field(\n        default_factory=list,\n        description="Dictionary mapping action names to Action objects"\n    )\n`,
		"action_only": `from pydantic import BaseModel, Field\n\nclass Action(BaseModel):\n    """\n    Represents a concrete physical action between entities in a narrative text.\n    """\n    source: str = Field(..., description="Name of the entity performing the action")\n    source_type: str = Field(..., description="Category of the source (person, animal, object, location)")\n    source_is_character: bool = Field(..., description="Whether the source is a named character")\n    target: str = Field(..., description="Name of the entity receiving the action")\n    target_type: str = Field(..., description="Category of the target (person, animal, object, location)")\n    target_is_character: bool = Field(..., description="Whether the target is a named character")\n    action: str = Field(..., description="The verb or short phrase describing the physical interaction")\n    consequence: str = Field(..., description="The immediate outcome or result of the action")\n    text_describing_the_action: str = Field(..., description="Text fragment describing the action exactly as it is written in the text")\n    text_describing_the_consequence: str = Field(..., description="Description of the consequence exactly as it is written in the text")\n    location: str = Field(..., description="location from global to local")\n    temporal_order_id: int = Field(..., description="Sequential identifier for chronological order")\n`
	};

	insertSchemaTemplateBtn.onclick = () => {
		const key = schemaTemplateSelect?.value || 'narrative_full';
		schemaInput.value = SCHEMA_TEMPLATES[key] || '';
		schemaStatus.textContent = 'Template inserted';
		log(`Inserted schema template: ${key}`);
	};

	clearSchemaBtn.onclick = () => {
		schemaInput.value = '';
		schemaStatus.textContent = '';
		log('Schema cleared.');
	};

	// Drag & drop upload
	const selectFile = () => fileInput.click();
	dropZone.onclick = selectFile;
	dropZone.ondragover = (e)=>{e.preventDefault(); dropZone.classList.add('drag');};
	dropZone.ondragleave = ()=>dropZone.classList.remove('drag');
	dropZone.ondrop = async (e)=>{
		e.preventDefault(); dropZone.classList.remove('drag');
		const file = e.dataTransfer?.files?.[0];
		if(file) await uploadParquet(file);
	};
	fileInput.onchange = async (e)=>{
		const file = e.target.files?.[0];
		if(file) await uploadParquet(file);
	};

	async function uploadParquet(file){
		const fd = new FormData();
		fd.append('file', file);
		log(`Uploading ${file.name}...`);
		const res = await fetch('/api/upload-parquet', {method:'POST', body: fd});
		const data = await res.json();
		if(!data.ok){ log(`Upload failed: ${data.error || 'unknown error'}`); return; }
		if(fileTitle) fileTitle.textContent = data.filename || '';
		fileMeta.textContent = `${data.filename} — rows: ${data.num_rows}, cols: ${data.num_cols}`;
		renderColumns(data.columns || []);
		expandedColumns = new Set();
		renderTable(data);
		log('Upload complete.');
	}

	// Category extraction buttons
	qsa('.chip').forEach(btn => {
		btn.onclick = async () => {
			const category = btn.dataset.category;
			const columns = Array.from(selectedColumns);
			const filters = filtersInput.value || '';
			log(`Starting extraction for ${category}...`);
			openProgressStream();
			const res = await postJSON('/api/extract', {category, columns, filters});
			if(res.ok){
				log(`Extraction finished: ${category}. ${res.result?.extracted?.length || 0} items.`);
				consolePanel.classList.add('glass');
			} else {
				log('Extraction failed.');
			}
		};
	});

	function openProgressStream(){
		try{
			const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/progress');
			ws.onmessage = (ev) => {
				try{ const msg = JSON.parse(ev.data); if(msg.type==='progress'){ log(`» ${msg.message}`); } }
				catch(e){ /* ignore */ }
			};
			ws.onclose = () => { log('Progress stream closed.'); };
		}catch(e){ log('WebSocket error.'); }
	}

	async function fetchGraphSlice(fileName, offset, length, mode, clean=false){
		const params = new URLSearchParams({ file: fileName, offset: String(offset), length: String(length), mode, clean: String(clean) });
		const url = `/api/embeddings-graph?${params.toString()}`;
		const res = await fetch(url);
		if(!res.ok){
			let msg = '';
			try{ const j = await res.json(); msg = j?.error || ''; }catch(_){ }
			log(`Fetch error ${res.status} ${res.statusText} → ${url}`);
			if(msg) log(`Server error: ${msg}`);
			return null;
		}
		const data = await res.json();
		if(!data?.ok){ log(`API error → ${url}`); return null; }
		return data;
	}

	async function buildSimpleGraph(modeOverride='interactions_all', doClean=false){
		if(!selectedModalFile){ log('Select a Modal file first.'); return; }
		const n = Math.min(500, Math.max(10, parseInt(rowsInput?.value || '200', 10)));
		simpleInfo && (simpleInfo.textContent = `building ${modeOverride.replace('_',' ')}${doClean?' (clean)':''} on first ${n} rows...`);
		resetGraph();
		const data = await fetchGraphSlice(selectedModalFile, 0, n, modeOverride, doClean);
		if(!data){ simpleInfo && (simpleInfo.textContent = 'build failed'); return; }
		for(const n of (data.nodes||[])) graphState.nodesById.set(n.id, { ...n });
		graphState.edges = data.edges || [];
		renderPlot();
		simpleInfo && (simpleInfo.textContent = `nodes ${graphState.nodesById.size}, edges ${graphState.edges.length}`);
	}

	let storyAbort = false;
	async function startStoryMode(){
		if(!selectedModalFile){ log('Select a Modal file first.'); return; }
		storyAbort = false;
		resetGraph();
		let len = 1;
		const maxLen = Math.min(500, Math.max(10, parseInt(rowsInput?.value || '200', 10)));
		simpleInfo && (simpleInfo.textContent = `story mode: 1 → ${maxLen}`);
		while(!storyAbort && len <= maxLen){
			const data = await fetchGraphSlice(selectedModalFile, 0, len, 'interactions', false);
			if(!data){ break; }
			for(const n of (data.nodes||[])) graphState.nodesById.set(n.id, { ...n });
			graphState.edges = data.edges || [];
			renderPlot();
			len += 1;
			await new Promise(r=>setTimeout(r, 30));
		}
		simpleInfo && (simpleInfo.textContent = storyAbort ? 'story mode: stopped' : 'story mode: done');
	}

	function stopStoryMode(){ storyAbort = true; }

	const dragRotate = qs('#dragRotate');
	const dragZoom = qs('#dragZoom');
	function setDragMode(mode){ try{ Plotly.relayout(graphPlot, {'scene.dragmode': mode}); }catch(e){} }
	if(dragRotate){ dragRotate.onclick = ()=> setDragMode('orbit'); }
	if(dragZoom){ dragZoom.onclick = ()=> setDragMode('turntable'); }

	log('Ready. Drop a parquet, select a file, and click Build Graph.');

	function getCurrentSchema(){ return schemaInput?.value || ''; }

	async function callModal(path, payload){
		log(`Calling Modal: ${path} ...`);
		try{
			const res = await postJSON(path, payload);
			if(res.ok){ log(`Modal OK: ${res.message || 'Success'}`); }
			else { log(`Modal ERROR`); }
		}catch(e){ log(`Modal request failed`); }
	}

	if(modalExtractSchemaBtn){
		modalExtractSchemaBtn.onclick = async () => {
			const schema = getCurrentSchema();
			if(!schema){ log('No schema to send.'); return; }
			await callModal('/api/modal/extract-schema', { schema });
		};
	}

	if(modalGenerateEmbeddingsBtn){
		modalGenerateEmbeddingsBtn.onclick = async () => {
			const schema = getCurrentSchema();
			await callModal('/api/modal/generate-embeddings', { schema });
		};
	}

	async function loadModalFiles(){
		try{
			const res = await fetch('/api/modal-files');
			const data = await res.json();
			if(!data.ok) return;
			const seen = new Set();
			const uniqueFiles = [];
			for(const f of (data.files || [])){
				if(!f || !f.name) continue;
				if(seen.has(f.name)) continue;
				seen.add(f.name);
				uniqueFiles.push(f);
			}
			function shortLabel(name){
				if(!name) return '';
				return name.length <= 44 ? name : (name.slice(0, 22) + '…' + name.slice(-21));
			}
			function formatSize(bytes){
				if(!Number.isFinite(bytes)) return '';
				const units = ['B','KB','MB','GB','TB'];
				let i = 0; let n = bytes;
				while(n >= 1024 && i < units.length - 1){ n /= 1024; i++; }
				const fixed = n >= 10 || i === 0 ? 0 : 1;
				return `${n.toFixed(fixed)} ${units[i]}`;
			}
			modalFilesNav.innerHTML = '';
			selectedModalFile = null;
			resetGraph();
			uniqueFiles.forEach((f) => {
				const btn = document.createElement('button');
				btn.className = 'chip';
				btn.textContent = shortLabel(f.name);
				btn.title = `${f.name} (${formatSize(f.size)})`;
				btn.onclick = () => {
					if(selectedModalFile === f.name) return;
					selectedModalFile = f.name;
					Array.from(modalFilesNav.querySelectorAll('.chip')).forEach(b => b.classList.remove('active'));
					btn.classList.add('active');
					log(`Selected Modal file: ${selectedModalFile}`);
				};
				modalFilesNav.appendChild(btn);
			});
			if(scanStatus){ scanStatus.textContent = `Found ${uniqueFiles.length} file(s)`; }
		}catch(e){ /* ignore */ }
	}

	function ensureResizeHandler(){
		if(graphState.resizeHandler) return;
		graphState.resizeHandler = () => {
			try{
				const update = { height: Math.max(200, graphPanel.clientHeight - 40) };
				Plotly.relayout(graphPlot, update);
				Plotly.Plots.resize(graphPlot);
			}catch(e){}
		};
		window.addEventListener('resize', graphState.resizeHandler);
	}

	if(scanModalBtn){
		scanModalBtn.onclick = async () => {
			if(scanStatus) scanStatus.textContent = 'Scanning...';
			await loadModalFiles();
		};
	}

	if(buildGraphBtn){
		buildGraphBtn.onclick = () => buildSimpleGraph('interactions_all');
	}
	if(buildCharsBtn){
		buildCharsBtn.onclick = () => buildSimpleGraph('interactions');
	}
	if(cleanViewBtn){
		cleanViewBtn.onclick = () => buildSimpleGraph('interactions', true);
	}
	if(storyBtn){ storyBtn.onclick = startStoryMode; }
	if(stopBtn){ stopBtn.onclick = stopStoryMode; }
})();


