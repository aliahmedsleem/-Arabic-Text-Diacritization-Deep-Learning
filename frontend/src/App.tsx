import { useState } from 'react';
import { BookOpen, Feather, Sparkles, Copy, RotateCcw, Check, AlertCircle, AlignRight, Info, ArrowLeft, ScrollText } from 'lucide-react';

const sample = 'العلم نور يضيء طريق الإنسان ويمنحه القدرة على فهم العالم من حوله';
const arabicMarks = /[\u064B-\u065F\u0670\u06D6-\u06ED]/g;
const countWords = (s: string) => s.trim() ? s.trim().split(/\s+/).length : 0;

export default function App() {
  const [input, setInput] = useState(sample);
  const [output, setOutput] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);
  const [tab, setTab] = useState<'editor'|'about'>('editor');
  async function restore() {
    if (!input.trim() || busy) return;
    setBusy(true); setError(''); setOutput('');
    try {
      const response = await fetch('/api/diacritize', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text:input})});
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'تعذر الاتصال بمحرك التشكيل');
      if (typeof data.diacritized_text !== 'string') throw new Error('استجابة غير صالحة من الخادم');
      setOutput(data.diacritized_text);
    } catch (e) { setError(e instanceof Error ? e.message : 'حدث خطأ غير متوقع'); }
    finally { setBusy(false); }
  }
  async function copyResult() {
    if (!output) return;
    try { await navigator.clipboard.writeText(output); setCopied(true); window.setTimeout(()=>setCopied(false),1800); }
    catch { setError('تعذر النسخ؛ تحقق من صلاحية الحافظة'); }
  }
  function reset() {setInput('');setOutput('');setError('');}
  return <div className="site-shell">
    <header className="topbar"><div className="topbar-inner"><a className="brand" href="/" aria-label="مِشْكَاة الرئيسية"><span className="brand-symbol">۞</span><span><strong>مِشْكَاة</strong><small>لِسَانُ العَرَبِيَّةِ</small></span></a><nav aria-label="القائمة الرئيسية"><button className={tab==='editor'?'active':''} onClick={()=>setTab('editor')}>المحرّر</button><button className={tab==='about'?'active':''} onClick={()=>setTab('about')}>عن المشروع</button></nav><span className="research-tag">مشروع بحثي · نسخة تجريبية</span></div></header>
    <main><section className="hero"><div className="ornament">❈ &nbsp; بِلُغَةِ الضَّادِ &nbsp; ❈</div><p className="eyebrow">نظام ذكي لفهم السياق اللغوي</p><h1>لِكُلِّ حَرْفٍ <em>حِكَايَةٌ</em></h1><p className="hero-desc">نحو تشكيلٍ عربيّ أدقّ، يحفظ المعنى ويستعيد جمال النصّ.<br/>منصّة بحثية تعتمد على التعلّم العميق لفهم الحركات في سياقها.</p><div className="hero-rule"><span>✦</span></div></section>
    {tab==='editor' ? <section className="workspace" aria-label="محرر التشكيل"><div className="workspace-title"><div><span className="section-kicker">المِحْبَرَةُ الرَّقْمِيَّةُ</span><h2>مُحَرِّرُ النُّصُوصِ</h2><p>أدخل النصّ العربي، ثم اطلب استعادة حركاته السياقية.</p></div><div className="status"><span className="status-dot"/> محرك النموذج قيد الإعداد</div></div>
      <div className="editor-grid"><article className="editor-card"><div className="editor-head"><div><AlignRight size={18}/><strong>النصّ الأصلي</strong></div><span>قبل التشكيل</span></div><textarea aria-label="النص الأصلي" dir="rtl" maxLength={20000} value={input} onChange={e=>{setInput(e.target.value);setOutput('');setError('');}} placeholder="اكتب هنا نصّك العربي..."/><div className="editor-foot"><span>{countWords(input)} كلمة · {input.length} حرف</span><button onClick={reset}><RotateCcw size={15}/> مسح النص</button></div></article>
      <article className="editor-card result"><div className="editor-head"><div><Sparkles size={18}/><strong>النصّ المُشَكَّل</strong></div><span>بعد المعالجة</span></div><div className={'output-area '+(!output?'placeholder':'')} dir="rtl" aria-live="polite">{output || 'سيظهر النصّ بعد تشغيل نموذج التشكيل المدرَّب...'}</div><div className="editor-foot"><span>{output ? output.match(arabicMarks)?.length || 0 : 0} حركة مستعادة</span><button disabled={!output} onClick={copyResult}>{copied?<Check size={15}/>:<Copy size={15}/>} {copied?'تم النسخ':'نسخ النتيجة'}</button></div></article></div>
      {error && <div className="notice error" role="alert"><AlertCircle size={19}/><span>{error} — النموذج لم يُجهَّز بعد؛ هذه الواجهة لا تعرض نتائج مُختلَقة.</span></div>}
      <div className="action-row"><p><Info size={17}/> لا تُعرض نتائج التشكيل إلا بعد توفر نموذج حقيقي على الخادم.</p><button className="primary-btn" onClick={restore} disabled={!input.trim()||busy}><Sparkles size={18}/>{busy?'جارٍ المعالجة...':'اسْتَعِدِ الحَرَكَاتِ'}<ArrowLeft size={17}/></button></div>
      <div className="feature-row"><div><BookOpen/><strong>فهم السياق</strong><span>تمييز الحركات بحسب معنى الجملة</span></div><div><Feather/><strong>جمال العربية</strong><span>محرّر عربي مهيّأ للنصوص الفصيحة</span></div><div><ScrollText/><strong>بحث قابل للقياس</strong><span>مقارنة النماذج وفق DER وWER</span></div></div></section> : <section className="about-panel"><span className="section-kicker">رِسَالَةُ المَشْرُوعِ</span><h2>التَّشْكِيلُ بَيْنَ اللُّغَةِ وَالذَّكَاءِ</h2><p>مِشْكَاة نظام بحثي قيد التطوير يهدف إلى استعادة الحركات العربية بالاعتماد على السياق باستخدام نماذج تعلم عميق. تُختبر جودة النتائج لاحقاً بمقاييس خطأ الحركات والكلمات، ولا تُنسب أي دقة للنظام قبل التدريب والتقييم المستقل.</p><button className="text-link" onClick={()=>setTab('editor')}>العودة إلى المحرّر ←</button></section>}
    </main><footer><span>مِشْكَاة · نُعِيدُ لِلْحُرُوفِ حَرَكَاتِهَا</span><span>منصّة بحثية للتشكيل العربي السياقي</span></footer></div>;
}
