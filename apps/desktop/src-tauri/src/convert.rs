//! 检测、分割、关键点项目转成多标签分类项目：每张图上标过的类别就是这张图的标签。
//! 新建一个项目，图片复制过去，原项目不动。

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use rusqlite::params;
use serde::Serialize;

use crate::annotations::{self, Annotation};
use crate::error::{msg, AppResult};
use crate::images;
use crate::project::Project;
use crate::t;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConvertResult {
    /// 新项目的文件夹
    pub path: String,
    /// 打上标签的图片数
    pub images: usize,
    /// 其中没有目标、打上「没有目标时用的类别」的图片数
    pub negatives: usize,
    /// 新项目的类别数
    pub labels: usize,
}

struct Src {
    path: PathBuf,
    split: String,
    labels: BTreeSet<i64>,
}

/// empty_label：标为「没有目标」的图打上这个类别（比如「全新」「正常」）；不填就不要这些图
pub fn to_multilabel(
    src: &Project,
    parent: &Path,
    name: &str,
    empty_label: Option<&str>,
    progress: &(dyn Fn(usize, usize) + Sync),
) -> AppResult<ConvertResult> {
    if crate::project::is_classify(&src.meta.task) {
        return Err(msg(t!("已经是分类项目了，不用转换")));
    }
    let empty_label = empty_label.map(str::trim).filter(|s| !s.is_empty());
    // 已标注、已审核的图，和每张图上标过的类别
    let mut items: BTreeMap<i64, Src> = BTreeMap::new();
    {
        let conn = src.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT i.id, i.path, i.split, a.label_id FROM images i
             LEFT JOIN annotations a ON a.image_id = i.id AND a.label_id IS NOT NULL
             WHERE i.status IN ('labeled', 'reviewed') ORDER BY i.id",
        )?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?, r.get::<_, Option<i64>>(3)?)))?;
        for row in rows {
            let (id, path, split, label) = row?;
            let it = items.entry(id).or_insert_with(|| Src { path: src.resolve(&path), split, labels: BTreeSet::new() });
            if let Some(l) = label {
                it.labels.insert(l);
            }
        }
    }
    if empty_label.is_none() {
        items.retain(|_, it| !it.labels.is_empty());
    }
    if items.is_empty() {
        return Err(msg(t!("还没有标过的图片，先标注一些再转换")));
    }

    let dst = Project::create(parent, name, "multilabel")?;
    // 类别：照原项目的顺序和颜色，只要用到的
    let used: BTreeSet<i64> = items.values().flat_map(|it| it.labels.iter().copied()).collect();
    let mut map: HashMap<i64, i64> = HashMap::new();
    for l in annotations::list_labels(src)? {
        if used.contains(&l.id) {
            map.insert(l.id, annotations::add_label(&dst, &l.name, &l.color)?.id);
        }
    }
    let empty_id = match empty_label {
        Some(n) if items.values().any(|it| it.labels.is_empty()) => {
            let existing = annotations::list_labels(&dst)?.into_iter().find(|l| l.name == n).map(|l| l.id);
            Some(match existing {
                Some(id) => id,
                None => annotations::add_label(&dst, n, "#8b949e")?.id,
            })
        }
        _ => None,
    };

    // 复制图片，再按原图对应写标签
    let paths: Vec<PathBuf> = items.values().map(|it| it.path.clone()).collect();
    let (_, ids) = images::import_mapped(&dst, &paths, true, progress)?;
    let mut done = 0;
    let mut negatives = 0;
    for it in items.values() {
        let Some(&id) = ids.get(&it.path) else { continue };
        let tags: Vec<i64> = if it.labels.is_empty() {
            negatives += 1;
            empty_id.into_iter().collect()
        } else {
            it.labels.iter().filter_map(|l| map.get(l).copied()).collect()
        };
        if tags.is_empty() {
            continue;
        }
        let anns: Vec<Annotation> = tags
            .into_iter()
            .map(|l| Annotation { label_id: Some(l), kind: "tag".into(), data: serde_json::json!([]), score: None, source: "manual".into() })
            .collect();
        annotations::save(&dst, id, &anns)?;
        // 训练集 / 验证集的划分照搬，两个项目的评估结果才好比
        if it.split != "unset" {
            dst.conn.lock().execute("UPDATE images SET split = ?2 WHERE id = ?1", params![id, it.split])?;
        }
        done += 1;
    }
    Ok(ConvertResult {
        path: dst.root.to_string_lossy().into_owned(),
        images: done,
        negatives,
        labels: annotations::list_labels(&dst)?.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::tests::{bbox, sample, temp};

    #[test]
    fn detect_to_multilabel() {
        let base = temp("convert");
        let p = sample(&base);
        // 第 9 张标成「没有目标」，第 10 张没标
        p.conn.lock().execute("UPDATE images SET status = 'reviewed' WHERE id = 9", []).unwrap();
        let car = annotations::list_labels(&p).unwrap()[0].id;
        annotations::save(&p, 2, &[bbox(car, 1.0, 1.0, 5.0, 5.0)]).unwrap();
        let out = base.join("out");
        std::fs::create_dir_all(&out).unwrap();

        let r = to_multilabel(&p, &out, "m", Some("全新"), &|_, _| {}).unwrap();
        assert_eq!((r.images, r.negatives, r.labels), (9, 1, 3));
        let q = Project::open(Path::new(&r.path)).unwrap();
        assert_eq!(q.meta.task, "multilabel");
        // 第 1 张有 car 和 person 两个标签，第 2 张只有 car
        let tags = |id: i64| annotations::load(&q, id).unwrap().annotations.iter().filter(|a| a.kind == "tag").count();
        let counts: Vec<usize> = (1..=9).map(tags).collect();
        assert_eq!(counts.iter().filter(|n| **n == 2).count(), 7);
        assert_eq!(counts.iter().filter(|n| **n == 1).count(), 2);

        // 不填「没有目标」的类别：那张图不要
        let r2 = to_multilabel(&p, &out, "m2", None, &|_, _| {}).unwrap();
        assert_eq!((r2.images, r2.negatives, r2.labels), (8, 0, 2));
        // 分类项目不能再转
        assert!(to_multilabel(&q, &out, "m3", None, &|_, _| {}).is_err());
        drop(p);
        drop(q);
        let _ = std::fs::remove_dir_all(base);
    }
}
