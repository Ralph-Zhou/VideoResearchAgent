"""Data structures for the Video DeepResearch task generation pipeline.

A VideoEntityGraph is a chain-shaped (one-hop-per-step) graph where each node
represents an entity and *some* nodes carry a linked video's frame-level
evidence.  The chain order is recorded in ``transition_path``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VideoFrameEvidence:
    """Caption of a single frame sampled from a video."""

    timestamp: float
    caption: str

    def to_dict(self) -> Dict[str, Any]:
        return {"timestamp": self.timestamp, "caption": self.caption}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VideoFrameEvidence":
        return cls(timestamp=float(d["timestamp"]), caption=str(d["caption"]))


@dataclass
class VideoInfo:
    """Metadata of the video linked to an entity."""

    video_id: str
    url: str
    title: str
    channel: Optional[str] = None
    duration: Optional[float] = None
    view_count: Optional[int] = None
    upload_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "url": self.url,
            "title": self.title,
            "channel": self.channel,
            "duration": self.duration,
            "view_count": self.view_count,
            "upload_date": self.upload_date,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VideoInfo":
        return cls(
            video_id=d["video_id"], url=d["url"], title=d.get("title", ""),
            channel=d.get("channel"), duration=d.get("duration"),
            view_count=d.get("view_count"), upload_date=d.get("upload_date"),
        )


@dataclass
class VideoEntity:
    """One hop in the chain: an entity plus the video that was mined for it.

    ``properties`` and ``relations`` are populated by Stage 2's entity-enrichment
    tail (text-search driven).  They mirror the entity_dict schema used by the
    pure-text browsecomp deep-research pipeline and let Stage 3 compose
    questions that narrow the target entity to a single unambiguous match.
    """

    name: str
    depth: int = 0                         # 0 = first real hop
    category: Optional[str] = None         # taxonomy tag from Stage 1
    video: Optional[VideoInfo] = None
    frames: List[VideoFrameEvidence] = field(default_factory=list)
    video_summary: Optional[str] = None    # VLM-generated synthesis of all frames
    next_entity: Optional[str] = None      # name chosen for the next hop
    next_entity_reason: Optional[str] = None
    properties: List[str] = field(default_factory=list)
    relations: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "depth": self.depth,
            "category": self.category,
            "video": self.video.to_dict() if self.video else None,
            "frames": [f.to_dict() for f in self.frames],
            "video_summary": self.video_summary,
            "next_entity": self.next_entity,
            "next_entity_reason": self.next_entity_reason,
            "properties": list(self.properties),
            "relations": dict(self.relations),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VideoEntity":
        props_raw = d.get("properties", [])
        rels_raw = d.get("relations", {})
        return cls(
            name=d["name"],
            depth=int(d.get("depth", 0)),
            category=d.get("category"),
            video=VideoInfo.from_dict(d["video"]) if d.get("video") else None,
            frames=[VideoFrameEvidence.from_dict(f) for f in d.get("frames", [])],
            video_summary=d.get("video_summary"),
            next_entity=d.get("next_entity"),
            next_entity_reason=d.get("next_entity_reason"),
            properties=[str(p) for p in props_raw if isinstance(p, str)],
            relations={
                str(k): str(v)
                for k, v in (rels_raw.items() if isinstance(rels_raw, dict) else [])
                if isinstance(k, str) and isinstance(v, str)
            },
        )


@dataclass
class VideoEntityGraph:
    """Ordered chain of VideoEntities forming one graph-construction trajectory."""

    entities: List[VideoEntity] = field(default_factory=list)

    # ── Mutation ──
    def add(self, entity: VideoEntity) -> None:
        self.entities.append(entity)

    # ── Query helpers ──
    @property
    def depth(self) -> int:
        return len(self.entities)

    @property
    def seed_name(self) -> str:
        return self.entities[0].name if self.entities else ""

    @property
    def transition_path(self) -> List[str]:
        return [e.name for e in self.entities]

    def to_dict(self) -> Dict[str, Any]:
        return {"entities": [e.to_dict() for e in self.entities]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VideoEntityGraph":
        g = cls()
        for ed in d.get("entities", []):
            g.entities.append(VideoEntity.from_dict(ed))
        return g

    # ── Rendering for downstream prompts ──
    def format_for_prompt(self, include_frames: bool = True, max_frames_per_hop: int = 12) -> str:
        """Human-readable string of the whole graph — fed to task-generation prompt."""
        blocks: List[str] = []
        for i, e in enumerate(self.entities):
            lines: List[str] = []
            lines.append(f"=== Hop {i} · Entity: {e.name} (depth={e.depth}, category={e.category or '-'}) ===")
            if e.video:
                lines.append(
                    f"Linked video: title={e.video.title!r}, url={e.video.url}, "
                    f"channel={e.video.channel}, duration={e.video.duration}s"
                )
            if e.video_summary:
                lines.append(f"Video synthesised description:\n{e.video_summary}")
            if e.properties:
                lines.append("Background properties (text-verified facts; safe to use as indirect descriptors):")
                for p in e.properties:
                    lines.append(f"  • {p}")
            if e.relations:
                lines.append("Named relations (entity → relationship):")
                for k, v in e.relations.items():
                    lines.append(f"  • {k}: {v}")
            if include_frames and e.frames:
                lines.append("Per-frame captions (subset):")
                shown = e.frames[:max_frames_per_hop]
                for f in shown:
                    lines.append(f"  • t={f.timestamp:.2f}s — {f.caption}")
                if len(e.frames) > max_frames_per_hop:
                    lines.append(f"  ...({len(e.frames) - max_frames_per_hop} more frames omitted)")
            if e.next_entity:
                lines.append(f"→ next entity picked: {e.next_entity} (reason: {e.next_entity_reason})")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)
