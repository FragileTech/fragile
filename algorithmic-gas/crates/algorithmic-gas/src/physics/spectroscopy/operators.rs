//! Built-in operator families. `ChannelSpec` is the operator: it implements
//! `LocalOperator` by matching its family, and every family keeps its algebra
//! in one file behind `signature` and `evaluate`. Operators outside the
//! families are injected through `Extensions::operator`.
mod baryon;
mod chirality;
mod dirac;
mod electroweak;
mod glueball;
mod meson;
mod tensor;
mod twistor;
mod vector;
use super::{
    config::ChannelSpec,
    contract::{
        Availability, Capabilities, Element, ElementKind, FrameState, LocalOperator,
        OperatorContext, Signature, channel_id,
    },
};
use crate::{GasConfig, GasError, Result, error::require};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

impl LocalOperator for ChannelSpec {
    fn id(&self) -> String {
        ChannelSpec::id(self)
    }
    fn signature(&self, kind: ElementKind, context: &OperatorContext<'_>) -> Result<Signature> {
        match self {
            Self::Meson { .. } => meson::signature(self, kind, context),
            Self::Vector { .. } => vector::signature(self, kind, context),
            Self::Baryon { .. } => baryon::signature(self, kind, context),
            Self::Glueball { .. } => glueball::signature(self, kind, context),
            Self::Tensor { .. } => tensor::signature(self, kind, context),
            Self::Dirac { .. } => dirac::signature(self, kind, context),
            Self::U1 { .. }
            | Self::Su2 { .. }
            | Self::ElectroweakMixed
            | Self::FitnessPhase
            | Self::CloneIndicator
            | Self::ParityVelocity { .. } => electroweak::signature(self, kind, context),
            Self::Chirality { .. } => chirality::signature(self, kind, context),
            Self::Twistor { .. } => twistor::signature(self, kind, context),
            Self::Custom { .. } => Err(GasError::Capability(
                "custom operators are injected, not configured".into(),
            )),
        }
    }
    fn evaluate(
        &self,
        element: &Element,
        state: &FrameState,
        context: &OperatorContext<'_>,
        out: &mut [f64],
    ) -> bool {
        match self {
            Self::Meson { .. } => meson::evaluate(self, element, state, context, out),
            Self::Vector { .. } => vector::evaluate(self, element, state, context, out),
            Self::Baryon { .. } => baryon::evaluate(self, element, state, context, out),
            Self::Glueball { .. } => glueball::evaluate(self, element, state, context, out),
            Self::Tensor { .. } => tensor::evaluate(self, element, state, context, out),
            Self::Dirac { .. } => dirac::evaluate(self, element, state, context, out),
            Self::U1 { .. }
            | Self::Su2 { .. }
            | Self::ElectroweakMixed
            | Self::FitnessPhase
            | Self::CloneIndicator
            | Self::ParityVelocity { .. } => {
                electroweak::evaluate(self, element, state, context, out)
            }
            Self::Chirality { .. } => chirality::evaluate(self, element, state, context, out),
            Self::Twistor { .. } => twistor::evaluate(self, element, state, context, out),
            Self::Custom { .. } => false,
        }
    }
}

/// An injected operator with the element kind it is measured on.
#[derive(Clone)]
pub struct Injected {
    pub kind: ElementKind,
    pub operator: Arc<dyn LocalOperator>,
}

/// One measured channel: an operator on one element kind, with its signature
/// when the operator exists in the context and the reason otherwise.
#[derive(Clone, Debug, PartialEq)]
pub struct Channel {
    pub id: String,
    /// `ChannelSpec::Custom` for an injected operator.
    pub spec: ChannelSpec,
    pub kind: ElementKind,
    pub availability: Availability,
    pub signature: Option<Signature>,
    /// Index into the injected operators; `None` for a built-in channel,
    /// whose operator is `spec` itself.
    pub injected: Option<usize>,
}
impl Channel {
    pub fn operator<'a>(&'a self, injected: &'a [Injected]) -> &'a dyn LocalOperator {
        match self.injected {
            Some(index) => injected[index].operator.as_ref(),
            None => &self.spec,
        }
    }
}
fn channel(
    operator: &dyn LocalOperator,
    spec: &ChannelSpec,
    kind: ElementKind,
    context: &OperatorContext<'_>,
) -> Result<Channel> {
    let (availability, signature) = match operator.signature(kind, context) {
        Ok(mut signature) => {
            if signature.components == 0 {
                return Err(GasError::Shape(format!(
                    "operator {} reports no components",
                    operator.id()
                )));
            }
            signature.requires = signature.requires.union(&kind.requires());
            let availability = context.capabilities.check(&signature.requires);
            (availability, Some(signature))
        }
        Err(GasError::Capability(reason)) => (Availability::unavailable(reason), None),
        Err(e) => return Err(e),
    };
    Ok(Channel {
        id: channel_id(&operator.id(), kind),
        spec: spec.clone(),
        kind,
        availability,
        signature,
        injected: None,
    })
}

/// Every channel of `context.measurement` in configuration order, then the
/// injected operators in their order. A channel is unavailable when its
/// operator does not exist in this context or the capabilities lack one of
/// its requirements; the others proceed.
pub fn channels(context: &OperatorContext<'_>, injected: &[Injected]) -> Result<Vec<Channel>> {
    let mut out = vec![];
    for spec in &context.measurement.channels {
        for kind in spec.kinds(context.measurement.pairs) {
            out.push(channel(spec, spec, kind, context)?);
        }
    }
    let mut ids: BTreeSet<String> = out.iter().map(|c| c.id.clone()).collect();
    for (index, entry) in injected.iter().enumerate() {
        let id = entry.operator.id();
        let spec = ChannelSpec::Custom {
            id: id.strip_prefix("custom/").unwrap_or_default().into(),
        };
        spec.validate()?;
        let mut channel = channel(entry.operator.as_ref(), &spec, entry.kind, context)?;
        require(
            ids.insert(channel.id.clone()),
            format!("duplicate injected channel {}", channel.id),
        )?;
        channel.injected = Some(index);
        out.push(channel);
    }
    Ok(out)
}

/// Catalog row: the algebra of a channel, never a particle name. A physical
/// name appears only as the optional `assignment` of an analysis configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogEntry {
    pub id: String,
    pub spec: ChannelSpec,
    pub kind: ElementKind,
    pub family: String,
    pub standard: bool,
    /// Availability in the context the catalog was built for.
    pub availability: Availability,
    /// Signature in `Capabilities::nominal()`; `None` when the operator does
    /// not exist even there.
    pub signature: Option<Signature>,
    pub assignment: Option<String>,
}

/// Rows for `specs` on every element kind of `context.measurement.pairs`.
pub fn catalog(
    context: &OperatorContext<'_>,
    specs: &[ChannelSpec],
    assignments: &BTreeMap<String, String>,
) -> Result<Vec<CatalogEntry>> {
    let (gas, capabilities) = (GasConfig::default(), Capabilities::nominal());
    let nominal = OperatorContext {
        gas: &gas,
        measurement: context.measurement,
        capabilities: &capabilities,
    };
    let standard = ChannelSpec::standard_set();
    let mut out = vec![];
    for spec in specs {
        for kind in spec.kinds(context.measurement.pairs) {
            let described = channel(spec, spec, kind, &nominal)?;
            let here = channel(spec, spec, kind, context)?;
            // Name the missing record even when the operator needs more than
            // this context offers to exist.
            let availability = match (&here.availability, &described.signature) {
                (Availability::Unavailable { .. }, Some(signature)) => context
                    .capabilities
                    .check(&signature.requires)
                    .and(here.availability),
                _ => here.availability,
            };
            out.push(CatalogEntry {
                assignment: assignments
                    .get(&described.id)
                    .or_else(|| assignments.get(&spec.id()))
                    .cloned(),
                id: described.id,
                spec: spec.clone(),
                kind,
                family: spec.family().into(),
                standard: standard.contains(spec),
                availability,
                signature: described.signature,
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::spectroscopy::{
        config::{ChiralityObservable, MeasurementConfig, PairSelection},
        contract::{ExchangeParity, Requirements},
    };
    struct Speed;
    impl LocalOperator for Speed {
        fn id(&self) -> String {
            "custom/speed".into()
        }
        fn signature(&self, _: ElementKind, _: &OperatorContext<'_>) -> Result<Signature> {
            Ok(Signature::new(
                Requirements::default(),
                1,
                ExchangeParity::Even,
            ))
        }
        fn evaluate(
            &self,
            element: &Element,
            state: &FrameState,
            _: &OperatorContext<'_>,
            out: &mut [f64],
        ) -> bool {
            let i = element.walkers[0] as usize;
            out[0] = state.x[i * state.d];
            state.eligible[i]
        }
    }
    #[test]
    fn walker_chiralities_live_on_sites_and_custom_specifications_measure_nothing() {
        let kinds_of =
            |observable| ChannelSpec::Chirality { observable }.kinds(PairSelection::Both);
        for observable in [ChiralityObservable::Chi, ChiralityObservable::LeftFraction] {
            assert_eq!(kinds_of(observable), vec![ElementKind::Site]);
        }
        assert_eq!(
            kinds_of(ChiralityObservable::LeftRightCoupling),
            vec![ElementKind::CloningPair]
        );
        let custom = ChannelSpec::Custom { id: "probe".into() };
        assert!(custom.kinds(PairSelection::Both).is_empty());
        let (gas, capabilities) = (GasConfig::default(), Capabilities::nominal());
        let measurement = MeasurementConfig {
            channels: vec![custom.clone()],
            ..MeasurementConfig::default()
        };
        let context = OperatorContext {
            gas: &gas,
            measurement: &measurement,
            capabilities: &capabilities,
        };
        assert!(matches!(
            custom.signature(ElementKind::Site, &context),
            Err(GasError::Capability(_))
        ));
        assert!(channels(&context, &[]).unwrap().is_empty());
    }
    #[test]
    fn injected_operators_follow_the_configured_channels_and_batch_evaluation_matches_the_oracle() {
        let (gas, capabilities) = (GasConfig::default(), Capabilities::nominal());
        let measurement = MeasurementConfig {
            channels: vec![],
            ..MeasurementConfig::default()
        };
        let context = OperatorContext {
            gas: &gas,
            measurement: &measurement,
            capabilities: &capabilities,
        };
        let injected = [Injected {
            kind: ElementKind::Site,
            operator: Arc::new(Speed),
        }];
        let listed = channels(&context, &injected).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, "custom/speed/site");
        assert_eq!(listed[0].spec, ChannelSpec::Custom { id: "speed".into() });
        assert!(listed[0].availability.is_available());
        assert!(matches!(
            channels(&context, &[injected[0].clone(), injected[0].clone()]),
            Err(GasError::Configuration(_))
        ));
        let state = FrameState {
            n: 2,
            d: 1,
            x: vec![0.5, -2.],
            eligible: vec![true, false],
            ..FrameState::default()
        };
        let site = |i: u32| Element {
            walkers: [i; 3],
            kind: ElementKind::Site,
            weight: 1.,
            generation: [0; 3],
        };
        let (mut values, mut valid) = (vec![0.; 2], vec![false; 2]);
        listed[0].operator(&injected).evaluate_all(
            &[site(0), site(1)],
            &state,
            &context,
            &mut values,
            &mut valid,
        );
        assert_eq!((values, valid), (vec![0.5, -2.], vec![true, false]));
    }
}
