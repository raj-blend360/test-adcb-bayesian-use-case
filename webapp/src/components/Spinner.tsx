import { cn } from '../lib/utils'

export default function Spinner({ className }: { className?: string }) {
  return (
    <div
      className={cn('animate-spin rounded-full border-2 border-gray-300 border-t-blue-600 h-5 w-5', className)}
    />
  )
}
