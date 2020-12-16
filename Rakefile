require 'html/proofer'

task :test do
  sh "bundle exec jekyll build"
  HTML::Proofer.new(
    "./_site",
    check_html: true,
    only_4xx: true
  ).run
end
