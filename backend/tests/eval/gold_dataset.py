"""
Gold Standard Benchmark Dataset for Arabic Multi-Speaker ASR and Diarization.
Contains authentic ground truth transcripts and annotated speaker turns across dialects.
"""
from typing import Dict, Any, List

GOLD_TEST_SET: Dict[str, Dict[str, Any]] = {
    "sample_meeting_45s": {
        "audio_path": "scratch_45s.wav",
        "description": "Levantine/Syrian conversational dialogue (3 speakers), fast turn-taking",
        "ground_truth_text": (
            "هلا اول شي نحكي عالاكل كل واحد اكيد بحياتو عم يمر بتجارب اكل يعني اكيد الواحد بياكل بحياتو مستحيل ما تجي حدا مثلا ما ياكل "
            "صح اب عاصم سوال اب كاسم كانو انت امبين عليك انو مراحل الاكل يلي بتمر فيها مراحل كتير "
            "كتير صعبة صح صحيح "
            "ومطبات صعبة يعني "
            "صحيح هلا المشكلة بالنسبة بالاكل ما بعرف شو بدي قلك هلا ان عندي مشكلة اذا تحليت بالصبر بسمن يعني فعليا "
            "انت يعني امتى حسيت حالك انو بلشت تسمن "
            "مقابلة الخير يا عرب بلشت حس حالي سمنت وقتها"
        ),
        "ground_truth_segments": [
            {
                "speaker": "SPEAKER_00",
                "start": 0.03,
                "end": 12.96,
                "text": "هلا اول شي نحكي عالاكل كل واحد اكيد بحياتو عم يمر بتجارب اكل يعني اكيد الواحد بياكل بحياتو مستحيل ما تجي حدا مثلا ما ياكل",
            },
            {
                "speaker": "SPEAKER_01",
                "start": 12.96,
                "end": 19.37,
                "text": "صح اب عاصم سوال اب كاسم كانو انت امبين عليك انو مراحل الاكل يلي بتمر فيها مراحل كتير",
            },
            {
                "speaker": "SPEAKER_00",
                "start": 19.37,
                "end": 20.75,
                "text": "كتير صعبة صح صحيح",
            },
            {
                "speaker": "SPEAKER_01",
                "start": 20.75,
                "end": 22.96,
                "text": "ومطبات صعبة يعني",
            },
            {
                "speaker": "SPEAKER_00",
                "start": 22.96,
                "end": 29.92,
                "text": "صحيح هلا المشكلة بالنسبة بالاكل ما بعرف شو بدي قلك هلا ان عندي مشكلة اذا تحليت بالصبر بسمن يعني فعليا",
            },
            {
                "speaker": "SPEAKER_02",
                "start": 29.92,
                "end": 33.92,
                "text": "انت يعني امتى حسيت حالك انو بلشت تسمن",
            },
            {
                "speaker": "SPEAKER_00",
                "start": 33.92,
                "end": 45.00,
                "text": "مقابلة الخير يا عرب بلشت حس حالي سمنت وقتها",
            },
        ],
    }
}
