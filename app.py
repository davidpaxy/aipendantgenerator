<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>NOVE25&RAE</title>
  <link rel="stylesheet" href="style2.css" />
</head>
<body>
  <!-- CANVAS -->
  <div id="canvasContainer" style="display:none;"><canvas id="mcanvas"></canvas></div>

  <!-- LOADER -->
  <div id="loader">
    <img id="loader-logo" src="https://dev.argentum.company/Ijewel3d/switch/logo.png" alt="Loader Image" />
    <div id="loader-bar"><div id="loader-bar-progress" style="width:0%"></div></div>
  </div>

  <!-- CONFIGURATOR PANEL -->
  <div id="mconfigurator"></div>

  <!-- TOP CONTROLS -->
  <div id="top-controls" style="display:none;">
    <div class="top-controls__item" id="share-btn"><img src="condividi.png" alt="Condividi" /></div>
    <div class="top-controls__item" id="auto-rotate-btn"><img src="auto.png" alt="Auto Rotate" /></div>
    <div class="top-controls__item" id="fullscreen-btn"><img src="full.png" alt="Fullscreen" /></div>
    <div class="top-controls__item" id="fit-camera-btn"><img src="fit.png" alt="Fit Camera" /></div>
  </div>

  <!-- SELECTION DISPLAY -->
  <div id="selection-display"></div>

  <script src="https://dist.pixotronics.com/webgi/runtime/bundle-0.14.0.js"></script>
  <script type="module">
    async function setupViewer() {
      const loader = document.getElementById('loader');
      const progressBar = document.getElementById('loader-bar-progress');
      const canvasContainer = document.getElementById('canvasContainer');
      const canvas = document.getElementById('mcanvas');

      canvasContainer.style.display = 'block';
      canvas.width = canvasContainer.clientWidth;
      canvas.height = canvasContainer.clientHeight;

      const viewer = new ViewerApp({ canvas });
      await addBasePlugins(viewer);
      await viewer.addPlugin(DiamondPlugin);

      const materialManager = viewer.getManager().materials;
      const materialCache = new Map();

      function clearCategory(category) {
        materialCache.forEach((mat, key) => {
          if (key.startsWith(category + '|')) materialManager.unregisterMaterial(mat);
        });
      }

      async function applyMaterial(url, category) {
        if (!materialCache.has(`${category}|${url}`)) {
          const loaded = await viewer.load(url);
          const mat = Array.isArray(loaded) ? loaded[0] : loaded;
          materialCache.set(`${category}|${url}`, mat);
          materialManager.applyMaterial(mat, category);
        } else {
          materialManager.applyMaterial(materialCache.get(`${category}|${url}`), category);
        }
      }

      viewer.renderer.setSize(canvas.width, canvas.height);
      viewer.scene.setDirty();
      (await viewer.getPlugin(GroundPlugin)).autoBakeShadows = false;

      let currentVariant = null;
      let selectedLetterName = '';
      let selectedMaterialName = 'argento';

      function updateSelectionDisplay() {
        document.getElementById('selection-display').textContent =
          `Lettera ${selectedLetterName} | ${selectedMaterialName}`;
      }

      const materials = [
        {
          name: 'argento',
          thumb: 'https://dev.argentum.company/Ijewel3d/switch/bianco.png',
          mats: {
            Metallo: 'https://dev.argentum.company/Ijewel3d/switch/bianco.pmat',
            brunitura: 'https://dev.argentum.company/Ijewel3d/switch/BrunituraBianco.pmat',
            sand: 'https://dev.argentum.company/Ijewel3d/switch/ArgentoSand.pmat',
            grezzo: 'https://dev.argentum.company/Ijewel3d/switch/ArgentoGrezzo.pmat'
          }
        },
        {
          name: 'oro',
          thumb: 'https://dev.argentum.company/Ijewel3d/switch/oro.png',
          mats: {
            Metallo: 'https://dev.argentum.company/Ijewel3d/switch/oro.pmat',
            brunitura: 'https://dev.argentum.company/Ijewel3d/switch/OroGrezzo.pmat',
            sand: 'https://dev.argentum.company/Ijewel3d/switch/OroSand.pmat',
            grezzo: 'https://dev.argentum.company/Ijewel3d/switch/OroSand.pmat'
          }
        },
        {
          name: 'rosa',
          thumb: 'https://dev.argentum.company/Ijewel3d/switch/rosa.png',
          mats: {
            Metallo: 'https://dev.argentum.company/Ijewel3d/switch/rosa.pmat',
            brunitura: 'https://dev.argentum.company/Ijewel3d/switch/RosaGrezzo.pmat',
            sand: 'https://dev.argentum.company/Ijewel3d/switch/RosaSand.pmat',
            grezzo: 'https://dev.argentum.company/Ijewel3d/switch/RosaSand.pmat'
          }
        },
        {
          name: 'nero',
          thumb: 'https://dev.argentum.company/Ijewel3d/switch/nero.png',
          mats: {
            Metallo: 'https://dev.argentum.company/Ijewel3d/switch/nero.pmat',
            brunitura: 'https://dev.argentum.company/Ijewel3d/switch/BrunituraBianco.pmat',
            sand: 'https://dev.argentum.company/Ijewel3d/switch/ArgentoSand.pmat',
            grezzo: 'https://dev.argentum.company/Ijewel3d/switch/neroSand.pmat'
          }
        }
      ];

      await viewer.addPlugin(new class extends SwitchNodeBasePlugin {
        static PluginType = 'SwitchNodePlugin';
        async _refreshUi() {
          if (!(await super._refreshUi())) return false;
          const cont = document.getElementById('mconfigurator');
          if (!cont.dataset.init) {
            cont.dataset.init = 'true';
            cont.innerHTML = '';

            // LETTER VARIANT PANEL
            const titleVar = document.createElement('div');
            titleVar.className = 'config-label';
            titleVar.textContent = 'MODELLI';
            const panelVar = document.createElement('div');
            panelVar.className = 'model-container';
            cont.append(titleVar, panelVar);

            const variants = [
              { name: '1', url: 'https://drive-weur-1.ijewel3d.com/files/1012c7_5236dc0490.glb', thumb: '1' },
              { name: '2', url: 'https://drive-weur-1.ijewel3d.com/files/2cdf11_19cd49fb32.glb', thumb: '2' },
              { name: '3', url: 'https://drive-weur-1.ijewel3d.com/files/32aaad_bab1d2fa72.glb', thumb: '3' },
              { name: '4', url: 'https://drive-weur-1.ijewel3d.com/files/4c0051_e61cf15bda.glb', thumb: '4' },
              { name: '5', url: 'https://drive-weur-1.ijewel3d.com/files/574e8b_8391cb84f6.glb', thumb: '5' },
              //{ name: 'FedinaModuli', url: 'https://drive-weur-1.ijewel3d.com/files/fedinamoduli_cd6d9b646c.glb', thumb: 'https://dev.argentum.company/Ijewel3d/switch/r.png' }
            ];

            variants.forEach(v => {
              const btn = document.createElement('div'); btn.className = 'model-square';
              const img = document.createElement('img'); img.src = v.thumb; img.alt = v.name; btn.append(img);

              btn.addEventListener('click', async () => {
                panelVar.querySelectorAll('.model-square').forEach(el => el.classList.remove('selected'));
                btn.classList.add('selected');
                if (selectedLetterName === v.name) return;

                // Visual feedback
                canvasContainer.style.opacity = '0.5';

                // Carica nuova lettera
                const asset = await viewer.load(v.url);
                const newVariant = Array.isArray(asset) ? asset[0]._modelObject : asset._modelObject;

                // Rimuovo tutti i precedenti
                viewer.scene.modelRoot.children.slice().forEach(ch => viewer.scene.modelRoot.remove(ch));

                // Aggiungo il nuovo
                viewer.scene.modelRoot.add(newVariant);
                viewer.fitToView(newVariant);
                currentVariant = newVariant;
                selectedLetterName = v.name;

                // Applico materiali
                const matObj = materials.find(m => m.name === selectedMaterialName);
                if (matObj) {
                  ['Metallo','brunitura','sand','grezzo'].forEach(clearCategory);
                  for (const [cat, url] of Object.entries(matObj.mats)) {
                    await applyMaterial(url, cat);
                  }
                  viewer.renderer.refreshPipeline();
                }

                updateSelectionDisplay();
                viewer.scene.setDirty({ sceneUpdate: true, frameFade: true });

                // Ripristino opacità
                canvasContainer.style.opacity = '1';
              });

              panelVar.append(btn);
            });

            // Selezione default A
            setTimeout(() => {
              const defaultBtn = panelVar.querySelector('img[alt="A"]').parentElement;
              if (defaultBtn) defaultBtn.click();
            }, 0);

            // MATERIALI PANEL
            const titleMat = document.createElement('div');
            titleMat.className = 'config-label'; titleMat.textContent = 'MATERIALI';
            const panelMat = document.createElement('div'); panelMat.className = 'material-circle-container';
            cont.append(titleMat, panelMat);

            materials.forEach(mat => {
              const btn = document.createElement('div'); btn.className = 'material-circle';
              const img = document.createElement('img'); img.src = mat.thumb; img.alt = mat.name; btn.append(img);

              btn.addEventListener('click', async () => {
                panelMat.querySelectorAll('.material-circle').forEach(el => el.classList.remove('selected'));
                btn.classList.add('selected');
                selectedMaterialName = mat.name;
                updateSelectionDisplay();
                ['Metallo','brunitura','sand','grezzo'].forEach(clearCategory);
                for (const [cat, url] of Object.entries(mat.mats)) {
                  await applyMaterial(url, cat);
                }
                viewer.renderer.refreshPipeline();
                viewer.scene.setDirty({ sceneUpdate: true, frameFade: true });
              });

              panelMat.append(btn);
            });

            // Default materiale
            setTimeout(() => {
              const defaultMat = panelMat.querySelector('img[alt="argento"]').parentElement;
              if (defaultMat) defaultMat.click();
            }, 0);

            // Accordion behavior
            [[titleVar,panelVar],[titleMat,panelMat]].forEach(([t,p]) => {
              t.addEventListener('click', () => {
                const was = p.classList.contains('active');
                cont.querySelectorAll('.model-container,.material-circle-container').forEach(el => el.classList.remove('active'));
                if (!was) p.classList.add('active');
              });
            });
          }
          return true;
        }
      });

      viewer.renderer.refreshPipeline();

      // Loader iniziale
      const assets = ['https://drive-weur-1.ijewel3d.com/files/1012c7_5236dc0490.glb'];
      for (let i = 0; i < assets.length; i++) {
        await viewer.load(assets[i]);
        progressBar.style.width = `${((i+1)/assets.length)*100}%`;
        viewer.scene.setDirty();
      }

      await new Promise(r => setTimeout(r,20));
      await (await viewer.getPlugin(GroundPlugin)).bakeShadows();
      loader.style.transition = 'opacity 0.5s ease';
      loader.style.opacity = '0';
      setTimeout(() => loader.style.display = 'none', 100);

      viewer.fitToView(viewer.scene.modelRoot);
      document.getElementById('top-controls').style.display = 'flex';

      // Eventi controlli
      const controls = viewer.scene.activeCamera.controls;
      document.getElementById('auto-rotate-btn').addEventListener('click', () => { controls.autoRotate = !controls.autoRotate; viewer.scene.setDirty(); });
      document.getElementById('fit-camera-btn').addEventListener('click', () => { viewer.fitToView(viewer.scene.modelRoot); viewer.scene.setDirty(); });
      document.getElementById('fullscreen-btn').addEventListener('click', () => { document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen(); });
      document.getElementById('share-btn').addEventListener('click', async () => {
        const shareData = { title: document.title, text: "Dai un'occhiata 3D!", url: window.location.href };
        try {
          if (navigator.share) await navigator.share(shareData);
          else if (navigator.clipboard) { await navigator.clipboard.writeText(shareData.url); alert('Link copiato!'); }
          else prompt('Copia il link:', shareData.url);
        } catch {}
      });
// Chiude i pannelli aperti quando clicchi sul canvas
canvas.addEventListener('click', () => {
  // rimuove la classe 'active' da entrambi i tipi di pannello
  document
    .querySelectorAll('.model-container.active, .material-circle-container.active')
    .forEach(panel => panel.classList.remove('active'));
});

      // Resize
      window.addEventListener('resize', () => {
        canvas.width = canvasContainer.clientWidth;
        canvas.height = canvasContainer.clientHeight;
        viewer.renderer.setSize(canvas.width, canvas.height);
        viewer.scene.setDirty();
      });
    }
    setupViewer();
  </script>
</body>
</html>
