//AI TECHNOLOGY SECTION
import { Graph, clock, shield } from '../assets'

export default function ScienceSection() {
  return (
    <section className="bg-[#0B0E1A] text-white py-16 px-6 md:px-12 lg:px-20">
      <h2 className="text-4xl md:text-5xl font-extrabold text-center mb-3">
        The Science Behind <span className="text-cyan-400">MindAI</span>
      </h2>
      <p className="text-center text-gray-400 text-base md:text-lg mb-14">
        Cutting-edge deep learning built for critical medical applications.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-7xl mx-auto">
        {/* Card 1 */}
        <div className="bg-[#13172A] border border-[#1E233F] rounded-2xl p-6 flex flex-col justify-between hover:scale-[1.02] transition-transform duration-300">
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg md:text-xl font-semibold">Detection Performance</h3>
              <span className="text-cyan-400 text-sm font-medium">Validation</span>
            </div>
            <img
              src={Graph}
              alt="Graph"
              className="rounded-lg w-full h-48 object-cover mb-5"
            />
            <p className="text-emerald-400 font-semibold text-xl mb-1">95%+</p>
            <p className="text-sm md:text-base text-gray-400 leading-relaxed">
              Sustained detection accuracy on diverse datasets.
            </p>
          </div>
        </div>

        {/* Card 2 */}
        <div className="bg-[#13172A] border border-[#1E233F] rounded-2xl p-6 flex flex-col justify-between hover:scale-[1.02] transition-transform duration-300">
          <div>
            <img src={clock} alt="Clock" className="w-10 h-10 mb-4" />
            <h3 className="text-lg md:text-xl font-semibold mb-2">Instant Speed</h3>
            <p className="text-sm md:text-base text-gray-400 leading-relaxed mb-4">
              Reduce typical analysis wait times from days to mere{" "}
              <span className="font-semibold text-white">minutes</span>. Critical insights when you need them most.
            </p>
            <p className="text-cyan-400 font-semibold text-lg">5 Minute Turnaround</p>
          </div>
        </div>

        {/* Card 3 */}
        <div className="bg-[#13172A] rounded-2xl p-6 flex flex-col justify-between hover:scale-[1.02] transition-transform duration-300">
          <div>
            <img src={shield} alt="Shield" className="w-10 h-10 mb-4" />
            <h3 className="text-lg md:text-xl font-semibold mb-2">Trusted Security</h3>
            <p className="text-sm md:text-base text-gray-400 leading-relaxed mb-4">
              Your privacy is paramount. All scan data is encrypted, anonymized,
              and handled in compliance with global data standards.
            </p>
            <p className="text-fuchsia-400 font-semibold text-lg">Encrypted & Private</p>
          </div>
        </div>
      </div>
    </section>
  );
}
