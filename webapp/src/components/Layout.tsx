import { NavLink, Outlet } from 'react-router-dom'
import {
  Upload,
  Sliders,
  BrainCircuit,
  BarChart3,
  Settings2,
  PieChart,
  TrendingUp,
} from 'lucide-react'
import { cn } from '../lib/utils'

const NAV = [
  { to: '/upload', label: 'Upload', Icon: Upload },
  { to: '/transform', label: 'Transform', Icon: Sliders },
  { to: '/model', label: 'Model', Icon: BrainCircuit },
  { to: '/results', label: 'Results', Icon: BarChart3 },
  { to: '/tune', label: 'Tune', Icon: Settings2 },
  { to: '/visualize', label: 'Visualize', Icon: PieChart },
  { to: '/optimize', label: 'Optimize', Icon: TrendingUp },
]

export default function Layout() {
  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <aside className="w-56 bg-gray-900 text-white flex flex-col">
        <div className="p-4 border-b border-gray-700">
          <h1 className="text-lg font-bold text-blue-400">Bayesian MMM</h1>
          <p className="text-xs text-gray-400 mt-0.5">Media Mix Modeller</p>
        </div>
        <nav className="flex-1 p-3 space-y-1">
          {NAV.map(({ to, label, Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white',
                )
              }
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  )
}
