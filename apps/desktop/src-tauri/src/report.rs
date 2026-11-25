//! 交付报告：把一次训练的数据、训练过程、评估结果、推荐阈值、误报测试整理成一个 HTML 文件。
//! 文件自带样式和图片（不依赖网络），用浏览器打开即可，需要 PDF 时直接「打印 → 另存为 PDF」。

use std::fs;
use std::path::{Path, PathBuf};

use crate::annotations;
use crate::error::{msg, AppResult};
use crate::project::Project;
use crate::training;
use crate::t;

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// 表头一行
fn ths(names: &[String]) -> String {
    let cells: String = names.iter().map(|n| format!("<th>{n}</th>")).collect();
    format!("<tr>{cells}</tr>")
}

fn pct(v: &serde_json::Value) -> String {
    v.as_f64().map(|x| format!("{:.1}%", x * 100.0)).unwrap_or_else(|| "—".into())
}

fn num(v: &serde_json::Value, digits: usize) -> String {
    v.as_f64().map(|x| format!("{x:.digits$}")).unwrap_or_else(|| "—".into())
}

fn fmt_time(ts: i64) -> String {
    chrono::DateTime::from_timestamp(ts, 0)
        .map(|t| t.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M").to_string())
        .unwrap_or_default()
}

/// 折线图（内联 SVG）：series = (名字, 颜色, 每个点的值)
fn chart(series: &[(&str, &str, Vec<Option<f64>>)], y_max: Option<f64>, x_labels: Option<&[String]>) -> String {
    let (w, h, l, r, t, b) = (640.0, 200.0, 44.0, 12.0, 10.0, 24.0);
    let n = series.iter().map(|s| s.2.len()).max().unwrap_or(0).max(1) as f64;
    let vals: Vec<f64> = series.iter().flat_map(|s| s.2.iter().flatten().copied()).filter(|v| v.is_finite()).collect();
    if vals.is_empty() {
        return format!("<p class=\"dim\">{}</p>", t!("没有数据"));
    }
    let lo = 0.0_f64.min(vals.iter().cloned().fold(f64::MAX, f64::min));
    let mut hi = y_max.unwrap_or_else(|| vals.iter().cloned().fold(f64::MIN, f64::max));
    if hi - lo < 1e-9 {
        hi = lo + 1.0;
    }
    let x = |i: f64| l + (i + 1.0) / n * (w - l - r);
    let y = |v: f64| t + (1.0 - (v - lo) / (hi - lo)) * (h - t - b);
    let mut s = format!("<svg viewBox=\"0 0 {w} {h}\" class=\"chart\">");
    for k in 0..5 {
        let v = lo + (hi - lo) * k as f64 / 4.0;
        s += &format!(
            "<line x1=\"{l}\" x2=\"{}\" y1=\"{y0:.1}\" y2=\"{y0:.1}\" class=\"grid\"/><text x=\"{}\" y=\"{:.1}\" class=\"tick\" text-anchor=\"end\">{v:.2}</text>",
            w - r,
            l - 6.0,
            y(v) + 4.0,
            y0 = y(v)
        );
    }
    let ticks: Vec<usize> = {
        let m = n as usize;
        let mut v = vec![0, m / 4, m / 2, (3 * m) / 4, m.saturating_sub(1)];
        v.dedup();
        v
    };
    for i in ticks {
        let label = x_labels.and_then(|ls| ls.get(i).cloned()).unwrap_or_else(|| (i + 1).to_string());
        s += &format!("<text x=\"{:.1}\" y=\"{}\" class=\"tick\" text-anchor=\"middle\">{}</text>", x(i as f64), h - 6.0, esc(&label));
    }
    for (_, color, v) in series {
        let pts: Vec<String> =
            v.iter().enumerate().filter_map(|(i, p)| p.filter(|p| p.is_finite()).map(|p| format!("{:.1},{:.1}", x(i as f64), y(p)))).collect();
        s += &format!("<polyline points=\"{}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2\"/>", pts.join(" "));
    }
    s += "</svg><div class=\"legend\">";
    for (name, color, _) in series {
        s += &format!("<span><i style=\"background:{color}\"></i>{}</span>", esc(name));
    }
    s + "</div>"
}

/// 生成报告并写到 out
pub fn export(p: &Project, run_id: i64, out: &Path) -> AppResult<()> {
    let detail = training::detail(p, run_id)?;
    let run = &detail.run;
    let c = &run.config;
    let task = p.meta.task.as_str();
    let cls = crate::project::is_classify(task);
    let (m50, m): (String, String) = match task {
        "classify" => (t!("准确率"), t!("宏平均 F1")),
        "multilabel" => ("mAP".into(), t!("宏平均 F1")),
        _ => ("mAP50".into(), "mAP50-95".into()),
    };
    let task_name = match task {
        "segment" => t!("实例分割"),
        "classify" => t!("图像分类"),
        "multilabel" => t!("多标签分类"),
        "pose" => t!("关键点"),
        _ => t!("目标检测"),
    };
    let model = if run.engine == training::EXTERNAL {
        let name = c["displayName"].as_str().map(String::from).unwrap_or_else(|| t!("外部模型"));
        t!("{name}（外部导入）", name = name)
    } else {
        c["model"].as_str().unwrap_or("").to_string()
    };
    let stats = p.stats()?;
    let labels = annotations::list_labels(p)?;
    let mut h = String::new();

    // ---------- 概览 ----------
    h += &format!(
        "<h1>{}</h1><p class=\"dim\">{}</p>",
        t!("{name} · 模型交付报告", name = esc(&p.meta.name)),
        t!("任务：{task} · 模型：{model} · 训练记录 {run} · 生成于 {time}",
            task = task_name,
            model = esc(&model),
            run = esc(&run.name),
            time = chrono::Local::now().format("%Y-%m-%d %H:%M")
        )
    );
    let ev = detail.eval.as_ref();
    let split_key = ev.and_then(|e| e["splits"].as_object()).and_then(|s| if s.contains_key("test") { Some("test") } else if s.contains_key("val") { Some("val") } else { None });
    let sp = split_key.and_then(|k| ev.map(|e| &e["splits"][k]));
    if let Some(sp) = sp {
        let which = if split_key == Some("test") { t!("测试集") } else { t!("验证集") };
        h += &format!(
            "<div class=\"kpis\"><div><b>{}</b><span>{m50}</span></div><div><b>{}</b><span>{m}</span></div><div><b>{}</b><span>{}</span></div><div><b>{}</b><span>{}</span></div></div><p class=\"dim\">{}</p>",
            if task == "classify" { pct(&sp["map50"]) } else { num(&sp["map50"], 3) },
            num(&sp["map"], 3),
            pct(&sp["precision"]),
            t!("精确率"),
            pct(&sp["recall"]),
            t!("召回率"),
            t!("以上为{which}（{n} 张图）上的结果。", which = which, n = sp["images"])
        );
    } else {
        h += &format!("<p class=\"warn\">{}</p>", t!("这次训练还没有评估结果，请在评估页点「用当前数据评估」后再导出报告。"));
    }

    // ---------- 数据 ----------
    h += &format!("<h2>{}</h2><table>", t!("数据"));
    h += &ths(&[t!("项目图片"), t!("已标注"), t!("已审核"), t!("跳过"), t!("训练 / 验证 / 测试"), t!("数据来源")]);
    let im = &c["images"];
    h += &format!(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{} / {} / {}</td><td>{}</td></tr></table>",
        stats.total,
        stats.labeled + stats.reviewed,
        stats.reviewed,
        stats.skipped,
        im["train"],
        im["val"],
        im["test"],
        esc(&c["versionName"].as_str().map(|v| t!("数据集版本「{v}」", v = v)).unwrap_or_else(|| t!("训练时的当前数据")))
    );
    h += &format!("<h3>{}</h3><table>", t!("各类别标注数量"));
    h += &ths(&[t!("类别"), t!("标注数")]);
    for l in &labels {
        h += &format!("<tr><td><i class=\"sw\" style=\"background:{}\"></i>{}</td><td>{}</td></tr>", esc(&l.color), esc(&l.name), l.count);
    }
    h += "</table>";

    // ---------- 训练 ----------
    h += &format!("<h2>{}</h2><table>", t!("训练"));
    let row = |k: &str, v: String| format!("<tr><th>{k}</th><td>{}</td></tr>", esc(&v));
    h += &row(&t!("模型"), model.clone());
    if run.engine != training::EXTERNAL {
        h += &row(&t!("训练轮数"), c["epochs"].to_string());
        h += &row(&t!("输入尺寸"), format!("{} px", c["imgSize"]));
        h += &row(&t!("批大小 / 学习率"), format!("{} / {}", c["batch"], c["lr"]));
        h += &row(&t!("起点"), c["baseRunName"].as_str().map(|n| t!("接着「{n}」继续训练", n = n)).unwrap_or_else(|| t!("预训练权重")));
        if let Some(b) = ev.and_then(|e| e["bestEpoch"].as_i64()) {
            h += &row(&t!("最好的一轮"), t!("第 {b} 轮", b = b));
        }
        if let Some(t) = detail.metrics.last().and_then(|m| m["time"].as_f64()) {
            h += &row(&t!("训练用时"), t!("{n} 分钟", n = format!("{:.0}", t / 60.0)));
        }
    }
    h += &row(&t!("训练时间"), fmt_time(run.created_at));
    h += "</table>";
    if !detail.metrics.is_empty() {
        let get = |k: &str| detail.metrics.iter().map(|m| m[k].as_f64()).collect::<Vec<_>>();
        h += &format!("<div class=\"two\"><div><h3>{}</h3>", t!("验证集指标"));
        h += &chart(&[(m50.as_str(), "#2fb67c", get("map50")), (m.as_str(), "#9b7bf2", get("map"))], Some(1.0), None);
        h += &format!("</div><div><h3>{}</h3>", t!("训练损失"));
        h += &chart(&[("loss", "#3d8bfd", get("loss"))], None, None);
        h += "</div></div>";
    }

    // ---------- 评估 ----------
    if let (Some(sp), Some(ev)) = (sp, ev) {
        h += &format!("<h2>{}</h2><h3>{}</h3><table>", t!("评估"), t!("各类别"));
        h += &ths(&[t!("类别"), if cls { t!("图片数") } else { t!("真实目标") }, t!("精确率"), t!("召回率"), if cls { "F1".into() } else { "AP50".into() }]);
        for pc in sp["perClass"].as_array().into_iter().flatten() {
            h += &format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                esc(pc["name"].as_str().unwrap_or("")),
                pc["gt"],
                pct(&pc["precision"]),
                pct(&pc["recall"]),
                num(&pc["ap50"], 3)
            );
        }
        h += "</table>";
        // 混淆矩阵
        let classes: Vec<String> = ev["classes"].as_array().into_iter().flatten().filter_map(|v| v.as_str().map(String::from)).collect();
        let mut names = classes.clone();
        if !cls {
            names.push(t!("背景"));
        }
        // 多标签分类没有混淆矩阵（空的）
        if let Some(cm) = sp["confusion"].as_array().filter(|c| !c.is_empty()) {
            h += &format!("<h3>{} <span class=\"dim\">{}</span></h3><table class=\"cm\"><tr><th></th>", t!("混淆矩阵"), t!("（行：真实，列：预测）"));
            for n in &names {
                h += &format!("<th>{}</th>", esc(n));
            }
            h += "</tr>";
            for (i, rowv) in cm.iter().enumerate() {
                h += &format!("<tr><th>{}</th>", esc(names.get(i).map(|s| s.as_str()).unwrap_or("")));
                for (j, v) in rowv.as_array().into_iter().flatten().enumerate() {
                    let n = v.as_i64().unwrap_or(0);
                    let cls_name = if n == 0 { "" } else if i == j { "ok" } else { "bad" };
                    h += &format!("<td class=\"{cls_name}\">{}</td>", if n == 0 { String::new() } else { n.to_string() });
                }
                h += "</tr>";
            }
            h += "</table>";
        }
        // 推荐阈值
        let th = &sp["thresholds"];
        if let Some(curve) = th["curve"].as_array() {
            let best = &th["best"];
            h += &format!(
                "<h2>{}</h2><p>{}</p>",
                t!("推荐置信度阈值"),
                t!("部署时建议阈值 <b>{conf}</b>：精确率 {p}、召回率 {r}（精确率和召回率兼顾，F1 最高）。误报多就调高，漏报多就调低。",
                    conf = num(&best["conf"], 2),
                    p = pct(&best["precision"]),
                    r = pct(&best["recall"])
                )
            );
            let labels_x: Vec<String> = curve.iter().map(|p| num(&p["conf"], 2)).collect();
            let g = |k: &str| curve.iter().map(|p| p[k].as_f64()).collect::<Vec<_>>();
            let (pr, rc) = (t!("精确率"), t!("召回率"));
            h += &chart(&[(pr.as_str(), "#3d8bfd", g("precision")), (rc.as_str(), "#e7a33e", g("recall")), ("F1", "#2fb67c", g("f1"))], Some(1.0), Some(&labels_x));
            let per: Vec<&serde_json::Value> = th["perClass"].as_array().into_iter().flatten().filter(|c| !c["conf"].is_null()).collect();
            if !per.is_empty() {
                h += &format!("<h3>{}</h3><table>", t!("各类别推荐阈值"));
                h += &ths(&[t!("类别"), t!("阈值"), t!("精确率"), t!("召回率")]);
                for c in per {
                    h += &format!(
                        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                        esc(c["name"].as_str().unwrap_or("")),
                        num(&c["conf"], 2),
                        pct(&c["precision"]),
                        pct(&c["recall"])
                    );
                }
                h += "</table>";
            }
        }
        // 错例样图
        let errs: Vec<i64> = sp["errors"].as_array().into_iter().flatten().filter_map(|e| e["imageId"].as_i64()).take(8).collect();
        let mut imgs = String::new();
        for id in errs {
            if let Ok((stored, hash)) = crate::images::locate(p, id) {
                if let Ok(t) = crate::images::ensure_thumb(p, &stored, &hash) {
                    if let Ok(bytes) = fs::read(&t) {
                        imgs += &format!("<img src=\"data:image/jpeg;base64,{}\"/>", crate::ai::b64(&bytes));
                    }
                }
            }
        }
        if !imgs.is_empty() {
            h += &format!(
                "<h3>{}</h3><p class=\"dim\">{}</p><div class=\"imgs\">{imgs}</div>",
                t!("错例样图"),
                t!("模型判断有误（漏检、误报或类别错）的部分图片，可作为后续补充数据的方向。")
            );
        }
    }

    // ---------- 误报测试 ----------
    if let Ok(t) = fs::read_to_string(PathBuf::from(&run.dir).join("fptest.json")) {
        if let Ok(f) = serde_json::from_str::<serde_json::Value>(&t) {
            let per = f["perClass"].as_object().map(|o| o.iter().map(|(k, v)| format!("{k} {v}")).collect::<Vec<_>>().join(t!("、").as_str())).unwrap_or_default();
            h += &format!(
                "<h2>{}</h2><p>{}{}</p>",
                t!("误报测试"),
                t!("用一段不应报警的视频（{source}）测试 {min} 分钟，阈值 {conf}：共报警 <b>{alarms}</b> 次，折合每小时 <b>{hour}</b> 次。",
                    source = esc(f["source"].as_str().unwrap_or("")),
                    min = format!("{:.0}", f["seconds"].as_f64().unwrap_or(0.0) / 60.0),
                    conf = num(&f["conf"], 2),
                    alarms = f["alarms"],
                    hour = f["perHour"]
                ),
                if per.is_empty() { String::new() } else { t!("按类别：{list}。", list = esc(&per)) }
            );
        }
    }

    h += &format!("<p class=\"foot\">{}</p>", t!("本报告由 Wulabel 生成。"));
    let lang = match crate::i18n::lang().as_str() {
        "zh" => "zh-CN".to_string(),
        l => l.to_string(),
    };
    let html = format!(
        "<!doctype html><html lang=\"{lang}\"><head><meta charset=\"utf-8\"><title>{}</title><style>{CSS}</style></head><body>{h}</body></html>",
        t!("{name} 模型交付报告", name = esc(&p.meta.name))
    );
    if let Some(dir) = out.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(out, html).map_err(|e| msg(t!("写报告失败：{e}", e = e)))?;
    Ok(())
}

const CSS: &str = r#"
body{font:14px/1.7 -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;color:#1f2329;max-width:960px;margin:32px auto;padding:0 24px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 8px;padding-bottom:4px;border-bottom:2px solid #e5e8ec}h3{font-size:15px;margin:16px 0 6px}
.dim{color:#7a828c;font-size:13px}.warn{color:#c2410c}.foot{margin-top:40px;color:#9aa1ab;font-size:12px}
table{border-collapse:collapse;width:100%;margin:6px 0;font-size:13px}th,td{border:1px solid #e5e8ec;padding:5px 8px;text-align:left}th{background:#f6f7f9;font-weight:600}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0 4px}.kpis div{background:#f6f8fb;border-radius:8px;padding:12px 14px}
.kpis b{display:block;font-size:26px}.kpis span{color:#7a828c;font-size:12px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}.chart{width:100%;height:auto}.chart .grid{stroke:#eceff3}.chart .tick{fill:#9aa1ab;font-size:11px}
.legend{font-size:12px;color:#5c6470}.legend span{margin-right:12px}.legend i,.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}
.cm td{text-align:center;min-width:28px}.cm td.ok{background:#dcf5e7}.cm td.bad{background:#fde2e0}
.imgs{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.imgs img{width:100%;border-radius:6px}
@media print{body{margin:0}h2{break-after:avoid}table,.imgs,.chart{break-inside:avoid}}
"#;
