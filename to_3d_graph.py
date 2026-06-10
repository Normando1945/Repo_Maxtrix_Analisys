import json, os, webbrowser
from collections import Counter

# Load graphify graph.json
in_path = os.path.join(os.path.dirname(__file__), "graphify-out", "graph.json")
with open(in_path) as f:
    g = json.load(f)

# Build color palette for communities
uniq_comm = sorted(set(n.get("community", -1) for n in g["nodes"]))
pal = ["#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F","#EDC948",
       "#B07AA1","#FF9DA7","#9C755F","#BAB0AC","#86BCB6","#D4A6C8",
       "#F1CE63","#A0CBE8","#FFBE7D","#FF9D9A","#8CD17D","#B6992D",
       "#499894","#D37295","#FABFD2","#D4A6C8","#9D7660","#D7B5A6"]
cc_map = {c: pal[i % len(pal)] for i, c in enumerate(uniq_comm)}

# Keep ALL nodes (no filtering)
nodes_out = []
for n in g["nodes"]:
    label = n.get("label", n["id"])
    nodes_out.append({
        "id": n["id"],
        "name": label,
        "community": n.get("community", -1),
        "color": cc_map.get(n.get("community", -1), "#888"),
        "file_type": n.get("file_type", ""),
        "source_file": n.get("source_file", ""),
        "degree": 0  # will fill below
    })

ids = {n["id"] for n in nodes_out}
node_color_map = {n["id"]: n["color"] for n in nodes_out}
links_out = []
for l in g["links"]:
    if l["source"] in ids and l["target"] in ids:
        links_out.append({
            "source": l["source"],
            "target": l["target"],
            "relation": l.get("relation", ""),
            "confidence": l.get("confidence", "EXTRACTED"),
            "color": node_color_map.get(l["source"], "#888")
        })

# Compute degree
deg = Counter()
for l in links_out:
    deg[l["source"]] += 1
    deg[l["target"]] += 1
for n in nodes_out:
    n["degree"] = deg.get(n["id"], 1)

# Build legend data — find the main node (highest degree) per community
comm_main = {}
for n in nodes_out:
    c = n["community"]
    if c not in comm_main or n["degree"] > comm_main[c]["degree"]:
        comm_main[c] = n

legend = []
seen = set()
for n in nodes_out:
    c = n["community"]
    if c not in seen:
        seen.add(c)
        cnt = sum(1 for x in nodes_out if x["community"] == c)
        main_name = comm_main[c]["name"]
        label = f"Community {c} / {main_name}"
        legend.append({"cid": c, "color": cc_map[c], "label": label, "count": cnt})
legend.sort(key=lambda x: x["cid"])

# Relation types for link width
rel_order = ["calls", "imports", "method", "contains", "references", "inherits", "re_exports", "rationale_for"]
rel_widths = {"calls": 1.0, "imports": 0.8, "method": 0.7, "contains": 0.6,
              "references": 0.5, "inherits": 0.9, "re_exports": 0.5, "rationale_for": 0.3}
rel_counts = Counter(l["relation"] for l in links_out)
rel_pal = ["#FFB74D","#81C784","#64B5F6","#E57373","#BA68C8","#4DD0E1","#FFF176","#A1887F"]
relation_legend = []
for i, r in enumerate(rel_order):
    cnt = rel_counts.get(r, 0)
    if cnt:
        relation_legend.append({"relation": r, "width": rel_widths[r], "count": cnt, "color": rel_pal[i % len(rel_pal)]})

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>3D Knowledge Graph</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0f0f1a; color:#e0e0e0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; margin:0; overflow:hidden; }}
#graph {{ position:fixed; top:0; left:0; width:100%; height:100%; }}
#sidebar {{ position:fixed; top:0; right:0; width:280px; height:100%; background:#1a1a2e; border-left:1px solid #2a2a4e; display:flex; flex-direction:column; overflow:hidden; z-index:100; }}
#credit-top {{ position:fixed; top:12px; left:14px; font-size:11px; color:rgba(255,255,255,0.45); z-index:99; user-select:none; pointer-events:none; line-height:1.4; }}
#credit-brand {{ font-size:9px; color:rgba(255,255,255,0.3); letter-spacing:1px; }}
#credit-btm {{ position:fixed; bottom:12px; left:14px; font-size:10px; color:rgba(255,255,255,0.35); z-index:99; user-select:none; pointer-events:none; }}
#search-wrap {{ padding:12px; border-bottom:1px solid #2a2a4e; position:relative; }}
#search {{ width:100%; background:#0f0f1a; border:1px solid #3a3a5e; color:#e0e0e0; padding:7px 10px; border-radius:6px; font-size:13px; outline:none; }}
#search:focus {{ border-color:#4E79A7; }}
#search-results {{ max-height:140px; overflow-y:auto; padding:4px 12px; border-bottom:1px solid #2a2a4e; display:none; }}
.search-item {{ padding:4px 6px; cursor:pointer; border-radius:4px; font-size:12px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.search-item:hover {{ background:#2a2a4e; }}
#info-panel {{ padding:14px; border-bottom:1px solid #2a2a4e; min-height:140px; }}
#info-panel h3 {{ font-size:13px; color:#aaa; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.05em; }}
#info-content {{ font-size:13px; color:#ccc; line-height:1.6; }}
#info-content .field {{ margin-bottom:5px; }}
#info-content .field b {{ color:#e0e0e0; }}
#info-content .empty {{ color:#555; font-style:italic; }}
.neighbor-link {{ display:block; padding:2px 6px; margin:2px 0; border-radius:3px; cursor:pointer; font-size:12px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; border-left:3px solid #333; }}
.neighbor-link:hover {{ background:#2a2a4e; }}
#neighbors-list {{ max-height:160px; overflow-y:auto; margin-top:4px; }}
#legend-wrap {{ flex:1; overflow-y:auto; padding:12px; }}
#legend-wrap h3 {{ font-size:13px; color:#aaa; margin-bottom:10px; text-transform:uppercase; letter-spacing:0.05em; }}
.legend-item {{ display:flex; align-items:center; gap:8px; padding:4px 0; cursor:pointer; border-radius:4px; font-size:12px; }}
.legend-item:hover {{ background:#2a2a4e; padding-left:4px; }}
.legend-item.dimmed {{ opacity:0.35; }}
.legend-dot {{ width:12px; height:12px; border-radius:50%; flex-shrink:0; }}
.legend-label {{ flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.legend-count {{ color:#666; font-size:11px; }}
#stats {{ padding:10px 14px; border-top:1px solid #2a2a4e; font-size:11px; color:#555; }}
#legend-controls {{ display:flex; align-items:center; gap:8px; margin-bottom:8px; padding:4px 0; }}
#legend-controls label {{ display:flex; align-items:center; gap:6px; cursor:pointer; font-size:12px; color:#aaa; user-select:none; }}
#legend-controls label:hover {{ color:#e0e0e0; }}
.legend-cb {{ appearance:none; width:14px; height:14px; border:1.5px solid #3a3a5e; border-radius:3px; background:#0f0f1a; cursor:pointer; flex-shrink:0; }}
.legend-cb:checked {{ background:#4E79A7; border-color:#4E79A7; }}
#rel-legend {{ position:fixed; bottom:14px; right:300px; background:rgba(15,15,26,0.85); border:1px solid #2a2a4e; border-radius:8px; padding:10px 14px; z-index:99; min-width:140px; }}
#rel-legend h4 {{ font-size:10px; color:#666; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.05em; }}
.rel-item {{ display:flex; align-items:center; gap:6px; padding:1px 0; font-size:10px; color:#999; }}
.rel-bar {{ height:3px; border-radius:2px; flex-shrink:0; }}
.rel-label {{ flex:1; white-space:nowrap; }}
</style>
</head>
<body>
<div id="graph"></div>
<div id="credit-top">MSc. Ing. Carlos Celi<br><span id="credit-brand">TORREFUERTE</span></div>
<div id="credit-btm">Data: graphify &bull; Viz: 3d-force-graph &bull; Built by Carlos Celi &amp; OpenCode</div>
<div id="rel-legend">
  <h4>Legend</h4>
  <div id="rel-legend-list"></div>
</div>
<div id="sidebar">
  <div id="search-wrap">
    <input id="search" type="text" placeholder="Search nodes..." autocomplete="off">
    <div id="search-results"></div>
  </div>
  <div id="info-panel">
    <h3>Node Info</h3>
    <div id="info-content"><span class="empty">Click a node to inspect it</span></div>
  </div>
  <div id="legend-wrap">
    <h3>Communities</h3>
    <div id="legend-controls">
      <label><input type="checkbox" class="legend-cb" id="select-all-cb" checked onchange="toggleAll(this.checked)">Select All</label>
    </div>
    <div id="legend"></div>
  </div>
  <div id="stats">{len(g["nodes"])} nodes &middot; {len(g["links"])} edges &middot; {len(uniq_comm)} communities</div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<script src="https://unpkg.com/3d-force-graph@1.76.3/dist/3d-force-graph.min.js"></script>
<script>
const NODES = {json.dumps(nodes_out)};
const EDGES = {json.dumps(links_out)};
const LEGEND = {json.dumps(legend)};
const REL_LEGEND = {json.dumps(relation_legend)};
const MAX_DEG = Math.max(...NODES.map(n => n.degree));
const REL_WIDTH = {{}};
REL_LEGEND.forEach(r => {{ REL_WIDTH[r.relation] = r.width; }});

// Build lookup maps
const nodeMap = {{}};
NODES.forEach(n => {{ nodeMap[n.id] = n; }});

// 3D Force Graph
const Graph = ForceGraph3D()(document.getElementById('graph'))
  .graphData({{ nodes: NODES, links: EDGES }})
  .nodeColor(n => n.color)
  .nodeRelSize(2.5)
  .nodeVal(n => 4 + 12 * (n.degree / MAX_DEG))
  .nodeLabel(n => n.name)
  .linkWidth(l => REL_WIDTH[l.relation] || 0.4)
  .linkDirectionalParticles(3)
  .linkDirectionalParticleSpeed(0.0035)
  .linkDirectionalParticleWidth(2)
  .linkDirectionalParticleColor(() => '#fff')
  .linkOpacity(0.3)
  .linkColor(l => l.color)
  .backgroundColor('#0f0f1a')
  .warmupTicks(100)
  .cooldownTicks(0)
  .onNodeClick(n => {{
    showNodeInfo(n);
    highlightNode(n);
  }})
  .onNodeHover(n => {{
    document.body.style.cursor = n ? 'pointer' : 'default';
  }});

function highlightNode(n) {{
  const gdata = Graph.graphData();
  if (!gdata || !gdata.links) return;
  const linkSet = new Set();
  gdata.links.forEach(l => {{
    const src = l.source && (l.source.id || l.source);
    const tgt = l.target && (l.target.id || l.target);
    if (src === n.id || tgt === n.id) linkSet.add(l);
  }});
  const neighborIds = new Set();
  linkSet.forEach(l => {{
    const src = l.source && (l.source.id || l.source);
    const tgt = l.target && (l.target.id || l.target);
    neighborIds.add(src);
    neighborIds.add(tgt);
  }});
  Graph.nodeColor(node =>
    node.id === n.id ? '#ff0' :
    neighborIds.has(node.id) ? node.color :
    '#333'
  );
  Graph.linkOpacity(l => linkSet.has(l) ? 0.8 : 0.05);
}}

// ---- Sidebar ----
function esc(s) {{
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}}

function showNodeInfo(n) {{
  const content = document.getElementById('info-content');
  let neighbors = [], uniqueNeighbors = [];
  try {{
    const gdata = Graph.graphData();
    if (gdata && gdata.links) {{
      neighbors = gdata.links.filter(l => (l.source && (l.source.id || l.source) === n.id)).map(l => l.target && (l.target.id || l.target))
        .concat(gdata.links.filter(l => (l.target && (l.target.id || l.target) === n.id)).map(l => l.source && (l.source.id || l.source)))
        .filter(Boolean);
      uniqueNeighbors = [...new Set(neighbors)];
    }}
  }} catch(e) {{ console.warn('showNodeInfo error', e); }}
  let html = '<div class="field"><b>Name:</b> ' + esc(n.name) + '</div>';
  html += '<div class="field"><b>File:</b> ' + esc(n.source_file || '-') + '</div>';
  html += '<div class="field"><b>Type:</b> ' + esc(n.file_type || '-') + '</div>';
  html += '<div class="field"><b>Community:</b> ' + esc(n.community) + '</div>';
  html += '<div class="field"><b>Connections:</b> ' + n.degree + '</div>';
  if (uniqueNeighbors.length) {{
    html += '<div style="margin-top:6px;font-size:12px;color:#aaa;">Neighbors:</div><div id="neighbors-list">';
    uniqueNeighbors.slice(0, 15).forEach(nid => {{
      const nn = nodeMap[nid];
      if (nn) html += '<div class="neighbor-link" onclick="focusNode(\\'' + nn.id + '\\')">' + esc(nn.name) + '</div>';
    }});
    if (uniqueNeighbors.length > 15) html += '<div style="color:#555;font-size:11px;margin-top:4px;">+' + (uniqueNeighbors.length - 15) + ' more</div>';
    html += '</div>';
  }}
  content.innerHTML = html;
}}

function focusNode(id) {{
  const n = nodeMap[id];
  if (!n) return;
  // NODES objects get x,y,z added in-place by d3-force after warmupTicks
  const x = n.x, y = n.y, z = n.z;
  if (x !== undefined) {{
    Graph.cameraPosition(
      {{x: x, y: y, z: 60}},  // camera position (z = distance)
      {{x: x, y: y, z: 0}},   // look-at target
      800                      // transition ms
    );
  }}
  setTimeout(() => showNodeInfo(n), 900);
}}

function showSearchResults(q) {{
  const el = document.getElementById('search-results');
  if (!q) {{ el.style.display = 'none'; return; }}
  const low = q.toLowerCase();
  const matches = NODES.filter(n => n.name.toLowerCase().includes(low) || n.id.toLowerCase().includes(low)).slice(0, 20);
  if (!matches.length) {{ el.style.display = 'none'; return; }}
  el.style.display = 'block';
  el.innerHTML = matches.map(n => '<div class="search-item" onclick="focusNode(\\'' + n.id + '\\')">' + esc(n.name) + '</div>').join('');
}}

// Link search input
document.getElementById('search').addEventListener('input', function() {{ showSearchResults(this.value); }});
document.getElementById('search').addEventListener('blur', function() {{ setTimeout(() => document.getElementById('search-results').style.display = 'none', 200); }});
document.getElementById('search').addEventListener('focus', function() {{ if (this.value) showSearchResults(this.value); }});

// Legend
function renderLegend() {{
  const el = document.getElementById('legend');
  el.innerHTML = LEGEND.map(l =>
    '<div class="legend-item" data-cid="' + l.cid + '" onclick="toggleCommunity(' + l.cid + ')">' +
      '<span class="legend-dot" style="background:' + l.color + '"></span>' +
      '<span class="legend-label">' + esc(l.label) + '</span>' +
      '<span class="legend-count">' + l.count + '</span>' +
    '</div>'
  ).join('');
}}

const visibleCommunities = new Set(LEGEND.map(l => l.cid));

function toggleCommunity(cid) {{
  if (visibleCommunities.has(cid)) visibleCommunities.delete(cid);
  else visibleCommunities.add(cid);
  applyFilter();
}}

function toggleAll(show) {{
  LEGEND.forEach(l => {{
    if (show) visibleCommunities.add(l.cid);
    else visibleCommunities.delete(l.cid);
  }});
  applyFilter();
}}

function applyFilter() {{
  // Update legend dimming
  document.querySelectorAll('.legend-item').forEach(el => {{
    const cid = parseInt(el.dataset.cid);
    el.classList.toggle('dimmed', !visibleCommunities.has(cid));
  }});
  // Show/hide nodes
  Graph.nodeVisibility(n => visibleCommunities.has(n.community));
  // Show/hide links — visible only if both ends have visible communities
  Graph.linkVisibility(l => {{
    const srcId = l.source && (l.source.id || l.source);
    const tgtId = l.target && (l.target.id || l.target);
    const srcC = nodeMap[srcId]?.community;
    const tgtC = nodeMap[tgtId]?.community;
    return visibleCommunities.has(srcC) && visibleCommunities.has(tgtC);
  }});
}}

function renderRelLegend() {{
  const el = document.getElementById('rel-legend-list');
  el.innerHTML = REL_LEGEND.map(r =>
    '<div class="rel-item"><div class="rel-bar" style="width:' + (10 + r.width * 20) + 'px;background:' + r.color + '"></div><span class="rel-label">' + r.relation + '</span><span class="rel-count">' + r.count + '</span></div>'
  ).join('');
}}

renderLegend();
renderRelLegend();
applyFilter();
</script>
</body>
</html>'''

out_path = os.path.join(os.path.dirname(__file__), "graph-3d.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Done: {out_path}  ({len(nodes_out)} nodes, {len(links_out)} edges)")
webbrowser.open(out_path)
