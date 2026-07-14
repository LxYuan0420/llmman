use std::path::PathBuf;

use clap::Args;

use crate::fmt::{human_size, relative_time, short_id};
use crate::storage::OciStore;

#[derive(Args, Debug)]
pub struct ListArgs {
    /// Local store directory (overrides default)
    #[arg(long, value_name = "DIR")]
    pub store: Option<PathBuf>,
}

pub fn run(args: &ListArgs) -> anyhow::Result<()> {
    let store_root = crate::default_store(args.store.as_deref())?;
    let store = OciStore::open(&store_root)?;
    let images = store.list()?;

    if images.is_empty() {
        return Ok(());
    }

    let name_w = images.iter().map(|i| i.reference.len()).max().unwrap_or(4).max(4);

    println!(
        "{:<name_w$}    {:<16}    {:<10}    {}",
        "NAME", "ID", "SIZE", "MODIFIED",
        name_w = name_w,
    );

    for img in &images {
        println!(
            "{:<name_w$}    {:<16}    {:<10}    {}",
            img.reference,
            short_id(&img.digest),
            human_size(img.size),
            relative_time(img.modified_at),
            name_w = name_w,
        );
    }
    Ok(())
}
