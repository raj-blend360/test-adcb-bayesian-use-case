import { cn } from '../lib/utils'

interface Props {
  status: string
}

const MAP: Record<string, { cls: string; label: string }> = {
  pending: { cls: 'badge-yellow', label: 'Pending' },
  running: { cls: 'badge-blue', label: 'Running' },
  complete: { cls: 'badge-green', label: 'Complete' },
  failed: { cls: 'badge-red', label: 'Failed' },
}

export default function StatusBadge({ status }: Props) {
  const { cls, label } = MAP[status] ?? { cls: 'badge', label: status }
  return <span className={cn('badge', cls)}>{label}</span>
}
