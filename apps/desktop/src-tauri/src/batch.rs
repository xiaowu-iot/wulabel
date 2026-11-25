//! 批量识别一个文件夹：用训练好的模型把文件夹里的图片都看一遍，
//! 结果导出成表格（Excel 能直接打开），也可以按识别出的类别把图片复制到各自的文件夹。

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};

use crate::dataset::safe_dir;
use crate::error::{msg, AppResult};
use crate::project::Project;
use crate::runtime::Runtime;
use crate::t;
use crate::training::{self, Predictor, RawPred};

static STOP: AtomicBool = AtomicBool::new(false);

/// 让正在跑的批量识别在这一批结束后停下（已经识别的照样写进表格）
pub fn stop() {
    STOP.store(true, Ordering::Relaxed);
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchOptions {
    pub run_id: i64,
    /// 要识别的文件夹（包括子文件夹）
    pub input: String,
    /// 结果放到这里
    pub output: String,
    pub conf: f64,
    /// 按类别把图片复制到各自的文件夹
    pub sort: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchResult {
    pub images: usize,
    /// 读不了、识别出错的图
    pub failed: usize,
    /// 表格文件
    pub csv: String,
    pub output: String,
    /// 每个类别的图片数（按图片数从多到少），最后是没有结果的
    pub counts: Vec<(String, usize)>,
    /// 拿不准（检测是没有目标）的图，可以加进项目去标
    pub uncertain: Vec<String>,
    pub stopped: bool,
}

/// 一张图识别出的类别：(类别序号, 最高分, 个数)。分类每个类别个数是 1
pub type Hits = Vec<(usize, f64, usize)>;

/// 把一张图的预测结果按任务整理成类别列表（按分数从高到低）。
/// 单标签分类只取最高的一个；多标签分类取分数够的全部；检测类数每个类别有几个目标
pub fn summarize(raw: &RawPred, task: &str, conf: f64) -> Hits {
    let mut hits: Hits = Vec::new();
    match task {
        "classify" => {
            if let Some(&(c, s)) = raw.tags.iter().max_by(|a, b| a.1.total_cmp(&b.1)) {
                if s >= conf {
                    hits.push((c, s, 1));
                }
            }
        }
        "multilabel" => {
            hits = raw.tags.iter().filter(|t| t.1 >= conf).map(|&(c, s)| (c, s, 1)).collect();
        }
        _ => {
            let objs = raw
                .boxes
                .iter()
                .map(|b| (b[4] as usize, b[5]))
                .chain(raw.polys.iter().map(|p| (p.0, p.1)))
                .chain(raw.poses.iter().map(|p| (p.0, p.1)));
            let mut by: BTreeMap<usize, (f64, usize)> = BTreeMap::new();
            for (c, s) in objs.filter(|o| o.1 >= conf) {
                let e = by.entry(c).or_insert((0.0, 0));
                e.0 = e.0.max(s);
                e.1 += 1;
            }
            hits = by.into_iter().map(|(c, (s, n))| (c, s, n)).collect();
        }
    }
    hits.sort_by(|a, b| b.1.total_cmp(&a.1));
    hits
}

/// 子文件夹里的图拍平成一个文件名：a/b/c.jpg → a_b_c.jpg
fn flat_name(input: &Path, file: &Path) -> String {
    let rel = file.strip_prefix(input).unwrap_or(file);
    let parts: Vec<String> = rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
    parts.join("_")
}

fn csv_field(s: &str) -> String {
    if s.contains(|c| matches!(c, ',' | '"' | '\n' | '\r')) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// 写表格、按类别复制图片。rows：每张图和它的识别结果（None 是识别出错）
pub fn write_outputs(
    input: &Path,
    out: &Path,
    classes: &[String],
    rows: &[(PathBuf, Option<Hits>)],
    sort: bool,
    detect: bool,
) -> AppResult<(PathBuf, Vec<(String, usize)>)> {
    fs::create_dir_all(out)?;
    let name_of = |c: usize| classes.get(c).cloned().unwrap_or_else(|| c.to_string());
    let none = if detect { t!("没有目标") } else { t!("拿不准") };
    // 表格：文件、结果、最高分，再每个类别一列（分类是分数，检测是个数）
    let mut csv = String::from("\u{feff}");
    let mut head = vec![t!("文件"), t!("结果"), t!("最高分")];
    head.extend(classes.iter().cloned());
    csv += &head.iter().map(|h| csv_field(h)).collect::<Vec<_>>().join(",");
    csv += "\r\n";
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut empty = 0;
    for (file, hits) in rows {
        let rel = file.strip_prefix(input).unwrap_or(file).to_string_lossy().replace('\\', "/");
        let mut line = vec![csv_field(&rel)];
        match hits {
            None => {
                line.push(csv_field(&t!("识别出错")));
                line.push(String::new());
                line.extend(classes.iter().map(|_| String::new()));
            }
            Some(h) => {
                let text = if h.is_empty() {
                    none.clone()
                } else if detect {
                    h.iter().map(|&(c, _, n)| format!("{}×{}", name_of(c), n)).collect::<Vec<_>>().join("；")
                } else {
                    h.iter().map(|&(c, _, _)| name_of(c)).collect::<Vec<_>>().join("；")
                };
                line.push(csv_field(&text));
                line.push(h.first().map(|x| format!("{:.3}", x.1)).unwrap_or_default());
                for c in 0..classes.len() {
                    line.push(match h.iter().find(|x| x.0 == c) {
                        Some(&(_, s, n)) => if detect { n.to_string() } else { format!("{s:.3}") },
                        None => String::new(),
                    });
                }
                if h.is_empty() {
                    empty += 1;
                }
                for &(c, _, _) in h {
                    *counts.entry(name_of(c)).or_default() += 1;
                }
                if sort {
                    let dirs: Vec<String> = if h.is_empty() { vec![none.clone()] } else { h.iter().map(|&(c, _, _)| name_of(c)).collect() };
                    for d in dirs {
                        let dir = out.join(safe_dir(&d));
                        fs::create_dir_all(&dir)?;
                        fs::copy(file, dir.join(flat_name(input, file)))
                            .map_err(|e| msg(t!("复制图片失败 {path}：{e}", path = file.display(), e = e)))?;
                    }
                }
            }
        }
        csv += &line.join(",");
        csv += "\r\n";
    }
    let path = out.join(format!("{}.csv", t!("识别结果")));
    fs::write(&path, csv)?;
    let mut list: Vec<(String, usize)> = counts.into_iter().collect();
    list.sort_by(|a, b| b.1.cmp(&a.1));
    if empty > 0 {
        list.push((none, empty));
    }
    Ok((path, list))
}

pub fn run(p: &Project, rt: &Runtime, pr: &Predictor, opts: &BatchOptions, progress: &(dyn Fn(usize, usize) + Sync)) -> AppResult<BatchResult> {
    STOP.store(false, Ordering::Relaxed);
    let input = PathBuf::from(&opts.input);
    let out = PathBuf::from(&opts.output);
    if !input.is_dir() {
        return Err(msg(t!("请选择要识别的文件夹")));
    }
    if opts.output.trim().is_empty() {
        return Err(msg(t!("请选择结果放在哪个文件夹")));
    }
    // 结果放在要识别的文件夹里面的话，下次再识别会把结果也算进去
    let canon = |p: &Path| fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    fs::create_dir_all(&out)?;
    if canon(&out).starts_with(canon(&input)) {
        return Err(msg(t!("结果文件夹不能放在要识别的文件夹里面")));
    }
    let files = crate::images::collect(&[input.clone()]);
    if files.is_empty() {
        return Err(msg(t!("这个文件夹里没有图片")));
    }
    let w = training::weights_of(p, opts.run_id)?;
    let task = p.meta.task.as_str();
    // 分类模型返回前几名，由这里按阈值取舍；检测类让推理进程直接按阈值过滤
    let req_conf = if crate::project::is_classify(task) { 0.01 } else { opts.conf };
    let items: Vec<(i64, PathBuf)> = files.iter().enumerate().map(|(i, f)| (i as i64, f.clone())).collect();
    let total = items.len();
    let mut results: Vec<Option<Hits>> = vec![None; total];
    let mut classes: Vec<String> = Vec::new();
    let mut done = 0;
    let mut stopped = false;
    progress(0, total);
    for chunk in items.chunks(16) {
        if STOP.load(Ordering::Relaxed) {
            stopped = true;
            break;
        }
        let mut got: Vec<(i64, RawPred)> = Vec::new();
        classes = pr.request(rt, &w, chunk, req_conf, &mut |id, r| got.push((id, r)))?;
        for (id, raw) in got {
            if let Some(slot) = results.get_mut(id as usize) {
                *slot = Some(summarize(&raw, task, opts.conf));
            }
        }
        done += chunk.len();
        progress(done, total);
    }
    let rows: Vec<(PathBuf, Option<Hits>)> = files.into_iter().zip(results).take(done).collect();
    let failed = rows.iter().filter(|r| r.1.is_none()).count();
    let uncertain = rows
        .iter()
        .filter(|r| r.1.as_ref().is_some_and(|h| h.is_empty()))
        .map(|r| r.0.to_string_lossy().into_owned())
        .collect();
    let (csv, counts) = write_outputs(&input, &out, &classes, &rows, opts.sort, !crate::project::is_classify(task))?;
    Ok(BatchResult {
        images: rows.len(),
        failed,
        csv: csv.to_string_lossy().into_owned(),
        output: out.to_string_lossy().into_owned(),
        counts,
        uncertain,
        stopped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::tests::temp;

    fn raw(tags: Vec<(usize, f64)>, boxes: Vec<[f64; 6]>) -> RawPred {
        RawPred { boxes, polys: Vec::new(), tags, poses: Vec::new() }
    }

    #[test]
    fn summarize_by_task() {
        let r = raw(vec![(1, 0.7), (0, 0.6), (2, 0.1)], Vec::new());
        assert_eq!(summarize(&r, "classify", 0.5), vec![(1, 0.7, 1)]);
        assert_eq!(summarize(&r, "classify", 0.8), vec![]);
        assert_eq!(summarize(&r, "multilabel", 0.5), vec![(1, 0.7, 1), (0, 0.6, 1)]);
        let d = raw(Vec::new(), vec![[0.0, 0.0, 1.0, 1.0, 0.0, 0.9], [0.0, 0.0, 1.0, 1.0, 0.0, 0.6], [0.0, 0.0, 1.0, 1.0, 1.0, 0.3]]);
        assert_eq!(summarize(&d, "detect", 0.5), vec![(0, 0.9, 2)]);
    }

    #[test]
    fn writes_csv_and_sorts() {
        let base = temp("batch");
        let input = base.join("in");
        fs::create_dir_all(input.join("sub")).unwrap();
        let a = input.join("a.jpg");
        let b = input.join("sub").join("b.jpg");
        let c = input.join("c.jpg");
        for f in [&a, &b, &c] {
            fs::write(f, b"x").unwrap();
        }
        let classes = vec!["扇叶损坏".to_string(), "灰尘, 污渍".to_string()];
        let rows = vec![(a.clone(), Some(vec![(0, 0.9, 1), (1, 0.7, 1)])), (b.clone(), Some(vec![])), (c.clone(), None)];
        let out = base.join("out");
        let (csv, counts) = write_outputs(&input, &out, &classes, &rows, true, false).unwrap();
        let text = fs::read_to_string(&csv).unwrap();
        let lines: Vec<&str> = text.trim_start_matches('\u{feff}').lines().collect();
        assert_eq!(lines.len(), 4);
        // 类别名里有逗号要加引号
        assert!(lines[0].ends_with(",扇叶损坏,\"灰尘, 污渍\""));
        assert!(lines[1].starts_with("a.jpg,"));
        assert!(lines[2].starts_with("sub/b.jpg,"));
        // 两个类别的图复制到两个文件夹，拿不准的放一起，子文件夹的图名字拍平
        assert!(out.join("扇叶损坏").join("a.jpg").exists());
        assert!(out.join(safe_dir("灰尘, 污渍")).join("a.jpg").exists());
        assert!(out.join(safe_dir(&t!("拿不准"))).join("sub_b.jpg").exists());
        assert_eq!(counts.len(), 3);
        let _ = fs::remove_dir_all(base);
    }
}
