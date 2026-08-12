from __future__ import annotations

import os

from orchestrator.agent import ComicAgent
from orchestrator.brain import DecisionBrain
from orchestrator.decision_engine import DecisionEngine
from orchestrator.state import JobState, JobStatus, PageStage, PageState, RegionState
from orchestrator.store import InMemoryStateStore
from orchestrator.tools import Tools

os.environ['MANGA_AGENT_LAMA_URL'] = 'http://127.0.0.1:9'
os.environ['MANGA_AGENT_LAMA_TIMEOUT'] = '1'

page = PageState(
    page_index=2,
    source_filename='page_0002.png',
    image_ref='data/test_runs/nightwing_vlm_live/page_0002/page.png',
    page_type='content',
    stage=PageStage.TRANSLATED,
    regions={
        1: RegionState(
            region_id=1,
            bbox=[136, 166, 341, 130],
            bbox_xyxy=[136, 166, 477, 296],
            translated_text='测试文字',
            has_text_box=True,
            color='black',
        )
    },
)
job = JobState(
    job_id='job-test-remote-erase',
    user_id='user',
    source_file_name='demo.cbz',
    target_language='zh',
    status=JobStatus.RUNNING,
    user_preferences={'pipeline_mode': 'vlm'},
    pages={2: page},
)

agent = ComicAgent(
    store=InMemoryStateStore(),
    tools=Tools(),
    decision_engine=DecisionEngine(),
    brain=DecisionBrain(),
)
agent._do_erase(job, page, retry=False, skill_params=None)

summary = agent._build_page_summary_for_brain(page)
strategy = agent.brain.suggest_page_strategy(
    page_summary=summary,
    pipeline_mode='vlm',
    target_language='zh',
    skill_context={},
)
next_action = agent.decision_engine.decide_next_action(job)

print({
    'page_stage': page.stage.value,
    'page_error_message': page.page_error_message,
    'page_error_stage': page.page_error_stage,
    'page_error_code': page.page_error_code,
    'erase_metrics': page.erase_metrics,
    'erase_qa': page.qa_reports.get('erase'),
    'brain_summary': strategy.get('summary'),
    'brain_strategy': strategy.get('strategy_suggestions'),
    'next_action': {
        'action_type': next_action.action_type.value,
        'reason': next_action.reason,
        'page_index': next_action.page_index,
    },
})
