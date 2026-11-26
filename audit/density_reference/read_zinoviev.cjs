// Read the supplied DjVu locally. DjVu.js 0.5.4 is downloaded from its author.
const fs = require("fs");
const vm = require("vm");
const path = require("path");
global.self = global;
global.document = {};
global.ImageData = class {
  constructor(a, b, c) {
    if (typeof a === "number") {
      this.width = a; this.height = b;
      this.data = new Uint8ClampedArray(a * b * 4);
    } else {
      this.data = a; this.width = b; this.height = c ?? a.length / (b * 4);
    }
  }
};
vm.runInThisContext(fs.readFileSync(path.join(__dirname, "djvu.js"), "utf8"));
const source = String.raw`C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project\TiAl-papers\density\Zinovjev_teplofizicheskie_svoistva_metallov.djvu`;
const buf = fs.readFileSync(source);
const doc = new DjVu.Document(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
(async () => {
  console.log("Pages:", doc.getPagesQuantity());
  for (const n of [219, 220, 221, 222, 223]) {
    const p = await doc.getPage(n);
    const t = p.getText();
    fs.writeFileSync(path.join(__dirname, `zinoviev_page_${n}.txt`), t);
    console.log(n, t.slice(0, 150), t.slice(-40));
    const img = p.getImageData();
    fs.writeFileSync(path.join(__dirname, `zinoviev_page_${n}.rgba`), img.data);
    fs.writeFileSync(path.join(__dirname, `zinoviev_page_${n}.json`),
      JSON.stringify({width: img.width, height: img.height, page: n}));
  }
})().catch(e => {console.error(e); process.exitCode = 1;});
