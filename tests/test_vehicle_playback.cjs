const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(process.argv[2],'utf8'),script=html.split('<script>')[1].split('</script>')[0];
const elements=new Map();let drawCalls=0;
const context=new Proxy({}, {get:(_,key)=>(...args)=>{if(['moveTo','lineTo','arc'].includes(key)){assert(args.every(Number.isFinite),`${key}: ${args}`);drawCalls++}}});
function el(id){if(!elements.has(id))elements.set(id,{value:'',checked:['compare'].includes(id),textContent:'',append(o){if(!this.value)this.value=o.value},getBoundingClientRect:()=>({width:800,height:id==='map'?480:240}),getContext:()=>context});return elements.get(id)}
el('rate').value=1;
const sandbox={document:{getElementById:el,createElement:()=>({})},window:{},devicePixelRatio:1,requestAnimationFrame:()=>{}};
vm.createContext(sandbox);vm.runInContext(script,sandbox);
assert(drawCalls>0);assert(el('readout').textContent.includes('Speed:'));
el('relative').checked=true;el('relative').onchange();assert.equal(el('seek').min,0);
el('vehicle').value='1';el('vehicle').onchange();el('compare').checked=false;el('compare').onchange();
el('seek').value=Number(el('seek').max);el('seek').oninput();el('play').onclick();assert.equal(el('play').textContent,'Pause');
console.log('Viewer initialization, drawing, comparison, time alignment, seek, play controls passed.');
