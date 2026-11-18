"""Reproducible, descriptive TV1 EDA. No model fitting or target imputation."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

WEEKDAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ nhật"]
FIGURE_NAMES = [
    "01_daily_sales.png", "02_weekday_sales.png", "03_sales_distribution.png",
    "04_family_sales_zero.png", "05_promotion_association.png", "06_example_series.png",
]


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def analyze(tables: dict[str, pd.DataFrame]) -> dict:
    """Compute summaries from all train records, preserving raw observations."""
    train, test = tables["train"], tables["test"]
    if train.empty or test.empty:
        raise ValueError("EDA requires nonempty train and test tables")
    sales = train["sales"]
    if sales.isna().any() or not np.isfinite(sales).all() or sales.lt(0).any():
        raise ValueError("EDA requires finite, observed, nonnegative train sales")
    family = train.assign(is_zero=sales.eq(0)).groupby("family", observed=True).agg(
        rows=("sales", "size"), total_sales=("sales", "sum"),
        mean_sales=("sales", "mean"), zero_rate=("is_zero", "mean"),
    ).reset_index().sort_values("total_sales", ascending=False)
    store = train.assign(is_positive=sales.gt(0)).groupby("store_nbr", observed=True).agg(
        rows=("sales", "size"), total_sales=("sales", "sum"),
        positive_rows=("is_positive", "sum"), observed_days=("date", "nunique"),
    ).reset_index().sort_values("total_sales", ascending=False)
    weekday = train.assign(dayofweek=train["date"].dt.dayofweek).groupby("dayofweek").agg(
        mean_sales=("sales", "mean"), rows=("sales", "size"),
    ).reset_index()
    weekday["weekday"] = weekday["dayofweek"].map(dict(enumerate(WEEKDAYS)))
    promo = train.assign(promoted=train["onpromotion"].gt(0))
    promo_overall = promo.groupby("promoted").agg(
        rows=("sales", "size"), mean_sales=("sales", "mean"), median_sales=("sales", "median"),
    ).reset_index()
    promo_family = promo.groupby(["family", "promoted"], observed=True).agg(
        rows=("sales", "size"), mean_sales=("sales", "mean"),
    ).reset_index()
    series = train.groupby(["store_nbr", "family"], observed=True)["sales"].agg(
        mean_sales="mean", zero_rate=lambda x: float(x.eq(0).mean())
    ).reset_index().sort_values(["mean_sales", "store_nbr", "family"])
    positive = series.loc[series["mean_sales"].gt(0)]
    pool = positive if len(positive) else series
    chosen = pool.iloc[sorted(set([0, len(pool) // 2, len(pool) - 1]))] if len(pool) else pool
    dates = pd.DatetimeIndex(train["date"].unique()).sort_values()
    missing = pd.date_range(dates.min(), dates.max()).difference(dates)
    quantiles = sales.quantile([0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "source": "7 unchanged CSV files in data/raw; sales summaries use train only",
        "train_rows": len(train), "test_rows": len(test),
        "train_start": str(train["date"].min().date()), "train_end": str(train["date"].max().date()),
        "test_start": str(test["date"].min().date()), "test_end": str(test["date"].max().date()),
        "test_days": int(test["date"].nunique()), "stores": int(train["store_nbr"].nunique()),
        "families": int(train["family"].nunique()), "series": len(series),
        "missing_train_dates": missing.strftime("%Y-%m-%d").tolist(),
        "sales": {
            "zero_rows": int(sales.eq(0).sum()), "zero_fraction": float(sales.eq(0).mean()),
            "fractional_rows": int(sales.mod(1).ne(0).sum()), "mean": float(sales.mean()),
            "median": float(quantiles.loc[0.5]), "p25": float(quantiles.loc[0.25]),
            "p75": float(quantiles.loc[0.75]), "p95": float(quantiles.loc[0.95]),
            "p99": float(quantiles.loc[0.99]), "max": float(sales.max()),
        },
        "family_summary": _records(family), "store_summary": _records(store),
        "weekday_summary": _records(weekday), "promotion_overall": _records(promo_overall),
        "promotion_by_family": _records(promo_family), "example_series": _records(chosen),
        "auxiliary": {
            "holiday_rows": len(tables["holidays_events"]),
            "holiday_dates": int(tables["holidays_events"]["date"].nunique()),
            "holiday_transferred_original_rows": int(tables["holidays_events"]["transferred"].astype(str).str.lower().eq("true").sum()),
            "oil_missing_values": int(tables["oil"]["dcoilwtico"].isna().sum()),
            "transactions_last_date": str(tables["transactions"]["date"].max().date()),
        },
    }


def create_figures(tables: dict[str, pd.DataFrame], analysis: dict, output_dir: Path) -> list[Path]:
    """Save six static, labeled figures. Matplotlib is imported only on demand."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    train = tables["train"]
    paths = [output_dir / name for name in FIGURE_NAMES]

    def save(fig, index):
        fig.savefig(paths[index], dpi=150, bbox_inches="tight")
        plt.close(fig)

    daily = train.groupby("date", observed=True)["sales"].sum().sort_index()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()))
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(daily.index, daily, color="#a9becb", lw=.7, label="Tổng sales quan sát")
    ax.plot(daily.index, daily.rolling("7D", min_periods=1).mean(), color="#176b87", lw=1.4,
            label="Trung bình 7 ngày lịch, bỏ qua ngày thiếu")
    ax.set(title="Diễn biến tổng sales theo ngày - tập train", xlabel="Ngày", ylabel="Tổng sales (đơn vị gốc hỗn hợp)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=.18)
    fig.tight_layout()
    save(fig, 0)

    week = pd.DataFrame(analysis["weekday_summary"]).set_index("dayofweek").reindex(range(7))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(WEEKDAYS, week["mean_sales"], color="#176b87")
    ax.set(title="Lượng bán trung bình mỗi dòng theo thứ - train", xlabel="Thứ", ylabel="Mean sales / dòng")
    ax.grid(axis="y", alpha=.18)
    fig.tight_layout()
    save(fig, 1)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.hist(np.log1p(train["sales"].to_numpy()), bins=70, color="#176b87", edgecolor="white", linewidth=.25)
    ax.set(title="Phân bố lượng bán trên thang log1p - toàn bộ train", xlabel="log1p(sales)", ylabel="Số dòng")
    ax.text(.98, .95, f"Sales = 0: {analysis['sales']['zero_fraction']:.2%}", transform=ax.transAxes,
            ha="right", va="top")
    fig.tight_layout()
    save(fig, 2)

    families = pd.DataFrame(analysis["family_summary"])
    top = families.head(12).sort_values("total_sales")
    zero = families.nlargest(12, "zero_rate").sort_values("zero_rate")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    axes[0].barh(top["family"], top["total_sales"], color="#176b87")
    axes[0].set(title="12 nhóm có tổng sales lớn nhất", xlabel="Tổng sales (đơn vị gốc hỗn hợp)")
    axes[1].barh(zero["family"], zero["zero_rate"] * 100, color="#d08b38")
    axes[1].set(title="12 nhóm có tỷ lệ sales = 0 cao nhất", xlabel="Tỷ lệ dòng bằng 0 (%)", xlim=(0, 105))
    fig.tight_layout(w_pad=3)
    save(fig, 3)

    overall = pd.DataFrame(analysis["promotion_overall"]).set_index("promoted").reindex([False, True])
    promo = pd.DataFrame(analysis["promotion_by_family"])
    chosen_names = families.head(10)["family"].tolist()
    paired = promo.pivot(index="family", columns="promoted", values="mean_sales").reindex(chosen_names)
    paired = paired.reindex(columns=[False, True])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7))
    axes[0].bar(["Không khuyến mãi", "Có khuyến mãi"], overall["mean_sales"], color=["#a9becb", "#176b87"])
    axes[0].set(title="So sánh quan sát gộp", ylabel="Mean sales / dòng")
    for i, promoted in enumerate([False, True]):
        if pd.notna(overall.loc[promoted, "rows"]):
            axes[0].text(i, float(overall.loc[promoted, "mean_sales"]),
                         f"n={int(overall.loc[promoted, 'rows']):,}", ha="center", va="bottom", fontsize=9)
    y = np.arange(len(paired))
    axes[1].barh(y-.18, paired[False], height=.35, label="Không khuyến mãi", color="#a9becb")
    axes[1].barh(y+.18, paired[True], height=.35, label="Có khuyến mãi", color="#176b87")
    axes[1].set_yticks(y, paired.index)
    axes[1].set(title="Trong 10 nhóm có tổng sales lớn nhất", xlabel="Mean sales / dòng (symlog)")
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.17), ncol=2)
    fig.suptitle("Khuyến mãi và lượng bán: liên hệ quan sát, không phải tác động nhân quả", fontsize=12)
    fig.tight_layout(w_pad=2.5)
    save(fig, 4)

    examples = analysis["example_series"]
    fig, axes = plt.subplots(max(1, len(examples)), 1, figsize=(12, 8), sharex=True, squeeze=False)
    start = train["date"].max() - pd.Timedelta(days=179)
    for ax, item in zip(axes[:, 0], examples):
        frame = train.loc[train["store_nbr"].eq(item["store_nbr"]) & train["family"].eq(item["family"])
                          & train["date"].ge(start)].sort_values("date")
        ax.plot(frame["date"], frame["sales"], color="#176b87", lw=1)
        ax.set(title=f"Cửa hàng {item['store_nbr']} - {item['family']}", ylabel="Sales (đơn vị của nhóm)")
        ax.set_ylim(bottom=0)
        if not frame.empty and frame["sales"].eq(0).all():
            ax.set_ylim(0, 1)
            ax.text(.5, .5, "180 ngày cuối đều có sales = 0", transform=ax.transAxes,
                    ha="center", va="center", color="#555555")
        ax.grid(axis="y", alpha=.18)
    axes[-1, 0].set_xlabel("Ngày (180 ngày lịch cuối train)")
    fig.suptitle("Ba chuỗi có phát sinh lượng bán: mean thấp, trung vị và cao", fontsize=12)
    fig.tight_layout()
    save(fig, 5)
    return paths


def write_eda_report(analysis: dict, paths: list[Path], report_dir: Path) -> None:
    """Write measured findings and limitations, without model prescriptions."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "eda_summary.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    sales, aux = analysis["sales"], analysis["auxiliary"]
    best_week = max(analysis["weekday_summary"], key=lambda row: row["mean_sales"])
    low_week = min(analysis["weekday_summary"], key=lambda row: row["mean_sales"])
    families = analysis["family_summary"]
    most_zero = max(families, key=lambda row: row["zero_rate"])
    promo = {row["promoted"]: row for row in analysis["promotion_overall"]}
    lines = [
        "# EDA - Phần việc TV1", "",
        "Nguồn: 7 CSV tại `data/raw/`. Các thống kê lượng bán chỉ dùng `train.csv`; không có mô hình được huấn luyện trong bước này.", "",
        "## Phạm vi dữ liệu", "",
        f"- Train: **{analysis['train_rows']:,} dòng**, {analysis['train_start']} đến {analysis['train_end']}.",
        f"- Test: **{analysis['test_rows']:,} dòng**, {analysis['test_days']} ngày, {analysis['test_start']} đến {analysis['test_end']}.",
        f"- Có **{analysis['stores']} cửa hàng, {analysis['families']} nhóm hàng, {analysis['series']:,} chuỗi cửa hàng-nhóm hàng**.",
        f"- Các ngày không xuất hiện trong train: {', '.join(analysis['missing_train_dates']) or 'không có'}. Ngày thiếu không được điền thành sales = 0.", "",
        "## Các phát hiện", "",
        f"1. **Nhiều số 0:** {sales['zero_rows']:,} dòng ({sales['zero_fraction']:.2%}) có sales = 0. Nhóm {most_zero['family']} có tỷ lệ cao nhất ({most_zero['zero_rate']:.2%}). Không thể chỉ từ sales = 0 kết luận hết hàng, đóng cửa hay không có nhu cầu.",
        f"2. **Phân bố lệch:** mean {sales['mean']:,.2f}, median {sales['median']:,.2f}, p95 {sales['p95']:,.2f}, p99 {sales['p99']:,.2f}, max {sales['max']:,.2f}. Cần xem thêm thang log để quan sát cả giá trị nhỏ và lớn; EDA không loại các giá trị lớn này.",
        f"3. **Khác biệt theo thứ:** {best_week['weekday']} có mean sales/dòng cao nhất ({best_week['mean_sales']:,.2f}); {low_week['weekday']} thấp nhất ({low_week['mean_sales']:,.2f}) khi gộp toàn bộ train. Kết quả mô tả không bảo đảm áp dụng giống nhau cho từng cửa hàng/nhóm hàng hoặc tương lai.",
        f"4. **Khác biệt quy mô nhóm hàng:** nhóm có tổng sales lớn nhất là {families[0]['family']}. Tổng này cộng đại lượng có thể khác đơn vị giữa các nhóm, nên không diễn giải thành doanh thu hoặc so sánh giá trị kinh tế.",
        f"5. **Đơn vị bán có thể lẻ:** có {sales['fractional_rows']:,} dòng sales là số thập phân; giữ kiểu số thực, không làm tròn nhãn thành số nguyên.",
    ]
    if False in promo and True in promo:
        lines += [f"6. **Liên hệ với khuyến mãi:** mean sales/dòng khi onpromotion > 0 là {promo[True]['mean_sales']:,.2f} (n={promo[True]['rows']:,}); khi bằng 0 là {promo[False]['mean_sales']:,.2f} (n={promo[False]['rows']:,}). Biểu đồ bổ sung so sánh trong từng family. Đây không phải ước lượng tác động: còn khác biệt thời gian, cửa hàng, nhu cầu và cách chọn khuyến mãi."]
    lines += ["", "## Bảng tóm tắt nhóm hàng", "", "| Nhóm hàng | Mean sales/dòng | Tỷ lệ sales = 0 |", "|---|---:|---:|"]
    lines += [f"| {row['family']} | {row['mean_sales']:,.2f} | {row['zero_rate']:.2%} |" for row in families]
    lines += ["", "## Dữ liệu phụ và quyết định bàn giao", "",
              f"- Ngày lễ: {aux['holiday_rows']} dòng nhưng chỉ {aux['holiday_dates']} ngày khác nhau; {aux['holiday_transferred_original_rows']} dòng là ngày gốc đã chuyển. Tổng hợp theo ngày/cửa hàng và phạm vi trước khi ghép, giữ cờ sự kiện riêng.",
              f"- Oil có {aux['oil_missing_values']} giá trị thiếu ngay trong CSV; lịch thiếu dòng được báo riêng trong data_quality.md. Bảng nền chưa tự thêm giá dầu hay nội suy.",
              f"- Transactions kết thúc {aux['transactions_last_date']}; không có giao dịch thực cho test. Giữ nguyên nguồn, không ghép giao dịch tương lai vào đầu vào dự báo.",
              "- Bảng nền giữ sales và id gốc, chỉ ghép metadata/lịch/lễ. Features lịch sử được TV1 tạo riêng bằng scripts/prepare_features.py; chủ model phụ trách encoding/scaling/chuỗi và huấn luyện theo mốc nhóm thống nhất.",
              "- Phân tích toàn bộ train ở đây phục vụ mô tả. Các kết luận liên quan lựa chọn đặc trưng không thay thế kiểm định dự báo theo thời gian.",
              "", "## Biểu đồ", ""]
    for path in paths:
        relative_path = Path(os.path.relpath(path, report_dir)).as_posix()
        lines += [f"![{path.stem}]({relative_path})", ""]
    (report_dir / "eda_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_eda(raw_dir: Path, figure_dir: Path, report_dir: Path) -> dict:
    from .data import load_raw_data
    tables = load_raw_data(raw_dir)
    analysis = analyze(tables)
    paths = create_figures(tables, analysis, figure_dir)
    write_eda_report(analysis, paths, report_dir)
    return analysis
