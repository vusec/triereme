use libafl::{corpus::CorpusScheduler, inputs::Input};

pub struct FaultCorpusScheduler;

impl<I, S> CorpusScheduler<I, S> for FaultCorpusScheduler
where
    I: Input,
{
    fn next(&self, _: &mut S) -> Result<usize, libafl::Error> {
        Err(libafl::Error::NotImplemented(
            "Scheduling attempted with fault scheduler".into(),
        ))
    }
}
