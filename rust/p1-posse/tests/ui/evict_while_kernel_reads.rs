use cutile::prelude::*;
use p1_posse::kernels::read_expert;
use p1_posse::SlotCache;

fn main() {
    let device = Device::new(0).unwrap();
    let stream = device.new_stream().unwrap();
    let mut cache = SlotCache::new(1, 1024, &stream).unwrap();
    let y = api::zeros::<f32>(&[1024]).sync_on(&stream).unwrap().partition([128]);
    let op = read_expert(y, cache.lease(0));
    cache.evict(0); // o kernel ainda segura o slot
    let _ = op.first().unpartition().to_host_vec().sync_on(&stream);
}
