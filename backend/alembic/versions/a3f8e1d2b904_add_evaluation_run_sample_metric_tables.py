"""add evaluation run sample metric tables

Revision ID: a3f8e1d2b904
Revises: 67546c96d522
Create Date: 2026-10-13 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a3f8e1d2b904'
down_revision: Union[str, None] = '67546c96d522'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SCHEMA = "component1"


def upgrade() -> None:
    # evaluation_runs
    op.create_table(
        'evaluation_runs',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('run_id', sa.String(length=64), nullable=False),
        sa.Column('registry_version', sa.String(length=16), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False, server_default='registered'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('notes', sa.Text(), nullable=False, server_default=''),

        # Protocol
        sa.Column('protocol_version', sa.String(length=16), nullable=False),
        sa.Column('conf_threshold', sa.Double(), nullable=False),
        sa.Column('iou_threshold_inference', sa.Double(), nullable=False),
        sa.Column('imgsz', sa.Integer(), nullable=False),
        sa.Column('iou_threshold_matching', sa.Double(), nullable=False),
        sa.Column('require_class_match', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('xai_methods', postgresql.JSONB(), nullable=False),
        sa.Column('seed', sa.Integer(), nullable=False, server_default='42'),

        # Manifest
        sa.Column('manifest_id', sa.String(length=64), nullable=False),
        sa.Column('source_manifest_hash', sa.String(length=128), nullable=False),
        sa.Column('split_plan_hash', sa.String(length=128), nullable=False),
        sa.Column('selected_sample_list_hash', sa.String(length=128), nullable=False),
        sa.Column('split', sa.String(length=8), nullable=False),
        sa.Column('selection_mode', sa.String(length=32), nullable=False),
        sa.Column('sample_count', sa.Integer(), nullable=False),

        # Model
        sa.Column('model_version', sa.String(length=64), nullable=False),
        sa.Column('model_weights_hash', sa.String(length=128), nullable=False),

        # Calibration
        sa.Column('calibration_status', sa.String(length=16), nullable=False, server_default='none'),
        sa.Column('calibration_version', sa.String(length=64), nullable=True),
        sa.Column('calibration_hash', sa.String(length=128), nullable=True),

        # Snapshots
        sa.Column('package_versions', postgresql.JSONB(), nullable=True),
        sa.Column('config_snapshot', postgresql.JSONB(), nullable=True),

        # Progress
        sa.Column('completed_samples', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('failed_samples', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_samples', sa.Integer(), nullable=False, server_default='0'),

        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('run_id'),
        sa.CheckConstraint(
            "status IN ('registered','running','completed','failed','cancelled')",
            name='chk_eval_run_status',
        ),
        sa.CheckConstraint(
            "calibration_status IN ('none','pending_qc','qc_validated')",
            name='chk_eval_calibration_status',
        ),
        sa.CheckConstraint(
            "selection_mode IN ('primary_representatives','all_views')",
            name='chk_eval_selection_mode',
        ),
        schema=SCHEMA,
    )
    op.create_index('ix_evaluation_runs_created_at', 'evaluation_runs', ['created_at'], schema=SCHEMA)
    op.create_index('ix_evaluation_runs_status', 'evaluation_runs', ['status'], schema=SCHEMA)

    # evaluation_samples
    op.create_table(
        'evaluation_samples',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('run_id', sa.Uuid(), nullable=False),
        sa.Column('sample_id', sa.String(length=64), nullable=False),
        sa.Column('filename', sa.String(length=256), nullable=False),
        sa.Column('source_key', sa.String(length=256), nullable=False),
        sa.Column('is_augmented', sa.Boolean(), nullable=False, server_default=sa.text('false')),
        sa.Column('source_unit_id', sa.String(length=256), nullable=False),
        sa.Column('confirmed_family_id', sa.String(length=256), nullable=False, server_default=''),
        sa.Column('assignment_group_root', sa.String(length=256), nullable=False),
        sa.Column('label_status', sa.String(length=16), nullable=False),
        sa.Column('annotation_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=False, server_default=''),

        # Detection counts
        sa.Column('detection_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('tp_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('fp_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('fn_count', sa.Integer(), nullable=False, server_default='0'),

        # Complexity
        sa.Column('complexity_luminance_var', sa.Double(), nullable=True),
        sa.Column('complexity_glcm_entropy', sa.Double(), nullable=True),
        sa.Column('complexity_colour_var', sa.Double(), nullable=True),

        # Timing
        sa.Column('detection_time_ms', sa.Double(), nullable=True),
        sa.Column('explanation_time_ms', sa.Double(), nullable=True),
        sa.Column('total_time_ms', sa.Double(), nullable=True),

        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),

        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['run_id'], [f'{SCHEMA}.evaluation_runs.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('run_id', 'sample_id', name='uq_eval_sample_run_sample'),
        sa.CheckConstraint(
            "status IN ('completed','failed','skipped')",
            name='chk_eval_sample_status',
        ),
        schema=SCHEMA,
    )
    op.create_index('ix_evaluation_samples_run_id', 'evaluation_samples', ['run_id'], schema=SCHEMA)
    op.create_index('ix_evaluation_samples_source_unit_id', 'evaluation_samples', ['source_unit_id'], schema=SCHEMA)
    op.create_index('ix_evaluation_samples_assignment_group_root', 'evaluation_samples', ['assignment_group_root'], schema=SCHEMA)

    # evaluation_detection_metrics
    op.create_table(
        'evaluation_detection_metrics',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('sample_id', sa.Uuid(), nullable=False),

        # Detection identity
        sa.Column('detection_order', sa.Integer(), nullable=False),
        sa.Column('detection_class_id', sa.SmallInteger(), nullable=False),
        sa.Column('detection_class_name', sa.String(length=32), nullable=False),
        sa.Column('detection_confidence', sa.Double(), nullable=False),
        sa.Column('detection_bbox_x_min', sa.Double(), nullable=False),
        sa.Column('detection_bbox_y_min', sa.Double(), nullable=False),
        sa.Column('detection_bbox_x_max', sa.Double(), nullable=False),
        sa.Column('detection_bbox_y_max', sa.Double(), nullable=False),
        sa.Column('detection_bbox_area', sa.Double(), nullable=False),

        # GT matching
        sa.Column('match_type', sa.String(length=4), nullable=False),
        sa.Column('match_iou', sa.Double(), nullable=True),
        sa.Column('matched_gt_idx', sa.Integer(), nullable=True),

        # XAI
        sa.Column('xai_method', sa.String(length=16), nullable=False),
        sa.Column('xai_outcome', sa.String(length=16), nullable=False),
        sa.Column('xai_error', sa.Text(), nullable=False, server_default=''),

        # Localization
        sa.Column('pointing_game', sa.Double(), nullable=True),
        sa.Column('energy_concentration', sa.Double(), nullable=True),

        # Faithfulness
        sa.Column('insertion_auc', sa.Double(), nullable=True),
        sa.Column('deletion_auc', sa.Double(), nullable=True),
        sa.Column('insertion_random_auc', sa.Double(), nullable=True),

        # Method agreement (paired, not duplicated per method)
        sa.Column('method_agreement_ssim', sa.Double(), nullable=True),

        # Diagnostics
        sa.Column('heatmap_nonzero_fraction', sa.Double(), nullable=True),
        sa.Column('heatmap_entropy', sa.Double(), nullable=True),

        # Timing
        sa.Column('generation_time_ms', sa.Double(), nullable=True),

        # Skip reasons
        sa.Column('metric_skip_reasons', postgresql.JSONB(), nullable=True),

        # Artifacts
        sa.Column('heatmap_artifact_path', sa.String(length=512), nullable=True),

        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),

        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['sample_id'], [f'{SCHEMA}.evaluation_samples.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('sample_id', 'detection_order', 'xai_method', name='uq_eval_det_metric_sample_det_method'),
        sa.CheckConstraint("match_type IN ('TP','FP','')", name='chk_eval_match_type'),
        sa.CheckConstraint(
            "xai_outcome IN ('success','degenerate','error','unavailable','ambiguous')",
            name='chk_eval_xai_outcome',
        ),
        schema=SCHEMA,
    )
    op.create_index('ix_eval_det_metrics_sample_id', 'evaluation_detection_metrics', ['sample_id'], schema=SCHEMA)
    op.create_index('ix_eval_det_metrics_class_id', 'evaluation_detection_metrics', ['detection_class_id'], schema=SCHEMA)


def downgrade() -> None:
    op.drop_table('evaluation_detection_metrics', schema=SCHEMA)
    op.drop_table('evaluation_samples', schema=SCHEMA)
    op.drop_table('evaluation_runs', schema=SCHEMA)
