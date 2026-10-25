"""Offline model smoke test: check that all six modules load and return a result."""
from domain import OPTIONS, TASKS, NUMERIC_BOUNDS, all_task_fields
from ml_service import ModelService


def sample_payload():
    payload = {'age': OPTIONS.get('age', ['[50-60)'])[5]}
    for task_id in TASKS:
        for field in all_task_fields(task_id):
            name = field['name']
            if name in payload:
                continue
            if field['type'] == 'number':
                payload[name] = str((field['min'] + field['max']) // 2)
            elif field['type'] == 'medication':
                payload[name] = 'Inconnu'
            else:
                opts = field.get('options') or []
                payload[name] = str(opts[0]) if opts else 'Inconnu'
    # Use plausible, in-range counts across all modules.
    payload.update({
        'num_lab_procedures':'40', 'num_procedures':'1', 'num_medications':'12',
        'number_outpatient':'0', 'number_emergency':'0', 'number_inpatient':'0', 'number_diagnoses':'6',
    })
    return payload


def main():
    service = ModelService()
    payload = sample_payload()
    numeric_unknown = dict(payload)
    for name in ('num_lab_procedures','num_procedures','num_medications','number_outpatient',
                 'number_emergency','number_inpatient','number_diagnoses'):
        numeric_unknown[name] = 'Inconnu'
    for task_id in TASKS:
        result = service.predict(task_id, payload)
        assert isinstance(result, dict) and result.get('label'), f'No result for {task_id}'
        print(f'[OK] {task_id}: {result["label"]}')
        result_unknown = service.predict(task_id, numeric_unknown)
        assert isinstance(result_unknown, dict) and result_unknown.get('label'), f'No result for {task_id} with unknown numeric fields'
        print(f'[OK unknown numeric] {task_id}')
    print('All six model smoke tests passed for synthetic documented/undocumented inputs. Software test only; no clinical validity implied.')


if __name__ == '__main__':
    main()
