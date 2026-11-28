//! 批量识别的结果文件：识别结果.csv（一个文件一行）、按单号汇总.csv（一个子文件夹一行）、
//! 识别报告.html（带缩略图，双击用浏览器打开），以及 .wulabel-records.json：
//! 记着每个文件识别过没有、大小和修改时间，勾上「跳过已识别过的」时没变的文件不再识别，结果沿用上次的。

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::AppResult;
use crate::t;

const RECORDS: &str = ".wulabel-records.json";

/// 一个文件（图片或视频）的识别结果
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Record {
    /// 相对要识别的文件夹，用 / 分隔
    pub rel: String,
    /// 所在子文件夹（比如单号），在最外层的是空的
    pub folder: String,
    pub video: bool,
    pub size: u64,
    pub mtime: u64,
    /// (类别名, 最高分, 个数)，视频的个数是检出的帧数；None 是识别出错
    pub hits: Option<Vec<(String, f64, usize)>>,
    /// 视频截图（绝对路径）
    #[serde(default)]
    pub shots: Vec<String>,
    /// 原文件（绝对路径），网页报告里图片的缩略图用
    pub path: String,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Saved {
    pub classes: Vec<String>,
    pub records: Vec<Record>,
}

/// 上次识别留下的记录（没有就是空的）
pub fn load(out: &Path) -> Saved {
    fs::read(out.join(RECORDS)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

/// 文件大小和修改时间（秒），判断上次识别后有没有变
pub fn stamp(path: &Path) -> (u64, u64) {
    let m = fs::metadata(path).ok();
    let size = m.as_ref().map(|m| m.len()).unwrap_or(0);
    let mtime = m
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    (size, mtime)
}

fn csv_field(s: &str) -> String {
    if s.contains(|c| matches!(c, ',' | '"' | '\n' | '\r')) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn csv_line(fields: &[String]) -> String {
    fields.iter().map(|f| csv_field(f)).collect::<Vec<_>>().join(",") + "\r\n"
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// 本机文件路径换成浏览器能打开的 file:/// 地址（中文、空格等都转义）
pub fn file_url(path: &str) -> String {
    let s = path.replace('\\', "/");
    let mut out = String::from(if s.starts_with('/') { "file://" } else { "file:///" });
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'/' | b'-' | b'_' | b'.' | b'~' | b':' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

pub struct Written {
    pub csv: PathBuf,
    pub summary: PathBuf,
    pub report: PathBuf,
}

/// 写全部结果文件。detect：检测类模型（类别列写个数）还是分类（写分数）
pub fn write_all(out: &Path, classes: &[String], records: &[Record], detect: bool) -> AppResult<Written> {
    fs::create_dir_all(out)?;
    let none = if detect { t!("没有目标") } else { t!("拿不准") };
    let text_of = |r: &Record| -> String {
        match &r.hits {
            None => t!("识别出错"),
            Some(h) if h.is_empty() => none.clone(),
            Some(h) if r.video => h.iter().map(|(c, _, n)| t!("{name}（{n} 帧）", name = c, n = n)).collect::<Vec<_>>().join("；"),
            Some(h) if detect => h.iter().map(|(c, _, n)| format!("{c}×{n}")).collect::<Vec<_>>().join("；"),
            Some(h) => h.iter().map(|(c, _, _)| c.clone()).collect::<Vec<_>>().join("；"),
        }
    };
    let top_of = |r: &Record| -> String {
        r.hits.as_ref().and_then(|h| h.iter().map(|x| x.1).reduce(f64::max)).map(|s| format!("{s:.3}")).unwrap_or_default()
    };

    // 1. 识别结果.csv：文件、所在文件夹、结果、最高分，再每个类别一列（分类是分数，检测是个数，视频检测是检出的帧数）
    let mut csv = String::from("\u{feff}");
    let mut head = vec![t!("文件"), t!("文件夹"), t!("结果"), t!("最高分")];
    head.extend(classes.iter().cloned());
    csv += &csv_line(&head);
    for r in records {
        let mut line = vec![r.rel.clone(), r.folder.clone(), text_of(r), top_of(r)];
        for c in classes {
            line.push(match r.hits.as_ref().and_then(|h| h.iter().find(|x| &x.0 == c)) {
                Some((_, s, n)) => if detect { n.to_string() } else { format!("{s:.3}") },
                None => String::new(),
            });
        }
        csv += &csv_line(&line);
    }
    let csv_path = out.join(format!("{}.csv", t!("识别结果")));
    fs::write(&csv_path, csv)?;

    // 2. 按单号汇总.csv：一个子文件夹（单号）一行；在最外层的文件各自一行
    let mut groups: BTreeMap<String, Vec<&Record>> = BTreeMap::new();
    for r in records {
        let key = if r.folder.is_empty() { r.rel.clone() } else { r.folder.clone() };
        groups.entry(key).or_default().push(r);
    }
    let mut sum = String::from("\u{feff}");
    sum += &csv_line(&[t!("单号（文件夹）"), t!("文件数"), t!("识别出的类别"), none.clone(), t!("识别出错")]);
    for (key, rs) in &groups {
        let mut by: BTreeMap<String, usize> = BTreeMap::new();
        for r in rs {
            for (c, _, _) in r.hits.iter().flatten() {
                *by.entry(c.clone()).or_default() += 1;
            }
        }
        let found = by.iter().map(|(c, n)| t!("{name}（{n} 个文件）", name = c, n = n)).collect::<Vec<_>>().join("；");
        let empty = rs.iter().filter(|r| r.hits.as_ref().is_some_and(|h| h.is_empty())).count();
        let failed = rs.iter().filter(|r| r.hits.is_none()).count();
        sum += &csv_line(&[key.clone(), rs.len().to_string(), found, empty.to_string(), failed.to_string()]);
    }
    let summary_path = out.join(format!("{}.csv", t!("按单号汇总")));
    fs::write(&summary_path, sum)?;

    // 3. 识别报告.html：按单号分组，每个文件一行，带缩略图
    let title = t!("识别报告");
    let mut html = format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width\"><title>{}</title><style>\
         body{{font:14px/1.5 system-ui,-apple-system,\"Microsoft YaHei\",sans-serif;margin:24px;color:#1f2329;background:#f6f7f9}}\
         h1{{font-size:22px;margin:0 0 4px}}.sub{{color:#6b727c;margin-bottom:20px}}\
         section{{background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:12px 16px;margin-bottom:14px}}\
         h2{{font-size:16px;margin:0 0 8px}}h2 small{{color:#6b727c;font-weight:400;margin-left:8px}}\
         table{{border-collapse:collapse;width:100%}}td{{border-top:1px solid #eef0f2;padding:8px 6px;vertical-align:top}}\
         td.th{{width:1%;white-space:nowrap}}td.th img{{height:96px;margin-right:6px;border-radius:6px;border:1px solid #e3e6ea}}\
         .none{{color:#b26a00}}.err{{color:#d03a33}}.f{{color:#6b727c;font-size:12px}}</style></head><body><h1>{}</h1><div class=\"sub\">{}</div>",
        esc(&title),
        esc(&title),
        esc(&t!("共 {n} 个文件，{g} 个单号（文件夹）", n = records.len(), g = groups.len())),
    );
    for (key, rs) in &groups {
        html += &format!("<section><h2>{}<small>{}</small></h2><table>", esc(key), esc(&t!("{n} 个文件", n = rs.len())));
        for r in rs {
            let imgs: Vec<String> = if r.video { r.shots.iter().take(4).cloned().collect() } else { vec![r.path.clone()] };
            let thumbs: String = imgs
                .iter()
                .map(|p| format!("<a href=\"{u}\" target=\"_blank\"><img loading=\"lazy\" src=\"{u}\"></a>", u = esc(&file_url(p))))
                .collect();
            let cls = match &r.hits {
                None => " class=\"err\"",
                Some(h) if h.is_empty() => " class=\"none\"",
                _ => "",
            };
            let name = r.rel.rsplit('/').next().unwrap_or(&r.rel);
            html += &format!(
                "<tr><td class=\"th\">{thumbs}</td><td><div{cls}>{}</div><div class=\"f\">{} · {}</div></td></tr>",
                esc(&text_of(r)),
                esc(name),
                esc(&top_of(r)),
            );
        }
        html += "</table></section>";
    }
    html += "</body></html>";
    let report_path = out.join(format!("{}.html", t!("识别报告")));
    fs::write(&report_path, html)?;

    // 4. 记录：下次「跳过已识别过的」用
    let saved = Saved { classes: classes.to_vec(), records: records.to_vec() };
    fs::write(out.join(RECORDS), serde_json::to_vec(&saved).map_err(|e| crate::error::msg(e.to_string()))?)?;
    Ok(Written { csv: csv_path, summary: summary_path, report: report_path })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(rel: &str, folder: &str, video: bool, hits: Option<Vec<(&str, f64, usize)>>) -> Record {
        Record {
            rel: rel.into(),
            folder: folder.into(),
            video,
            size: 1,
            mtime: 2,
            hits: hits.map(|h| h.into_iter().map(|(c, s, n)| (c.to_string(), s, n)).collect()),
            shots: if video { vec![format!("C:\\结果\\{folder}\\扇叶损坏 0.93.jpg")] } else { Vec::new() },
            path: format!("D:\\in\\{rel}"),
        }
    }

    #[test]
    fn writes_tables_report_and_records() {
        let out = std::env::temp_dir().join(format!("wulabel-report-{}", std::process::id()));
        let _ = fs::remove_dir_all(&out);
        let classes = vec!["扇叶损坏".to_string(), "灰尘, 污渍".to_string()];
        let records = vec![
            rec("SF1/a.mp4", "SF1", true, Some(vec![("扇叶损坏", 0.93, 3)])),
            rec("SF1/b.mp4", "SF1", true, Some(vec![])),
            rec("SF2/c.jpg", "SF2", false, None),
            rec("d.jpg", "", false, Some(vec![("扇叶损坏", 0.8, 1), ("灰尘, 污渍", 0.6, 2)])),
        ];
        let w = write_all(&out, &classes, &records, true).unwrap();
        let csv = fs::read_to_string(&w.csv).unwrap();
        let lines: Vec<&str> = csv.trim_start_matches('\u{feff}').lines().collect();
        assert_eq!(lines.len(), 5);
        assert!(lines[0].ends_with(",扇叶损坏,\"灰尘, 污渍\""));
        assert!(lines[1].starts_with("SF1/a.mp4,SF1,"));
        // 按单号汇总：SF1 两个文件，一个识别出扇叶损坏，一个什么都没有
        let sum = fs::read_to_string(&w.summary).unwrap();
        let sl: Vec<&str> = sum.trim_start_matches('\u{feff}').lines().collect();
        assert_eq!(sl.len(), 4, "SF1、SF2、d.jpg 各一行");
        let sf1 = sl.iter().find(|l| l.starts_with("SF1,")).unwrap();
        assert!(sf1.starts_with("SF1,2,") && sf1.ends_with(",1,0"), "{sf1}");
        // 网页报告：中文路径转义成 file:/// 地址
        let html = fs::read_to_string(&w.report).unwrap();
        assert!(html.contains("file:///C:/%E7%BB%93%E6%9E%9C/SF1/"));
        assert!(html.contains("%200.93.jpg"));
        // 记录能读回来
        let saved = load(&out);
        assert_eq!(saved.classes, classes);
        assert_eq!(saved.records, records);
        let _ = fs::remove_dir_all(out);
    }

    #[test]
    fn file_urls() {
        assert_eq!(file_url("D:\\a b\\c.jpg"), "file:///D:/a%20b/c.jpg");
        assert_eq!(file_url("/home/u/x#1.jpg"), "file:///home/u/x%231.jpg");
    }
}
